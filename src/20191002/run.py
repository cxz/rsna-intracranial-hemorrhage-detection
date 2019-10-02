
import sys

sys.path.insert(0, '.')

import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50

import util

#notes: added augment_prob=0.5, normalize with read_dicom3, weighted log loss, mixed

config = {
    'train_path': '../data/stage_1_train_images_jpg/',
    'train_path_dicom': '../data/stage_1_train_images',
    'test_path': '../data/stage_1_test_images_jpg/',
    'batch_size': 64,
    'workers': 8,
    'image_size': (224, 224),
    'augment_prob': 0.5,
    'experiment': '20191001-7',
    'note': '...',
    'data_path': '../data'
}


def run(config, resume_training=False):
        
    experiment_path = util.prepare_experiment(config)
    print(f"running {experiment_path}")
    
    #test_df = util.read_testset()
    df = util.read_trainset()

    # train set (90%) and validation set (10%)
    from sklearn.model_selection import ShuffleSplit
    ss = ShuffleSplit(n_splits=5, test_size=0.1, random_state=42).split(df.index)

    # will just do one fold
    train_idx, valid_idx = next(ss)

    print(f"train: {len(train_idx)}, val: {len(valid_idx)}")
    print(f"train_steps: {len(train_idx)//config['batch_size']}")

    gen = util.BalancedTrainDataGenerator(
        df.iloc[train_idx], 
        config['batch_size'], 
        config['image_size'], 
        augment_prob=config['augment_prob'], 
        #img_dir=config['train_path_dicom'], 
        #img_ext='dcm'
        img_dir=config['train_path'], 
        img_ext='jpg'
    )
    
    val_gen = util.RsnaDataGenerator(
        df.iloc[valid_idx], 
        config['batch_size'], 
        config['image_size'], 
        augment_prob=None,
        #img_dir=config['train_path_dicom'],
        #img_ext='dcm'
        img_dir=config['train_path'],
        img_ext='jpg'
    )
    
    model = util.Model1(
        engine=ResNet50,        
        #engine=tf.keras.applications.InceptionResNetV2,
        input_dims=config['image_size'], batch_size=config['batch_size'],         
        weights="imagenet", 
        verbose=1).model

    ckpt_name = os.path.join(experiment_path, 'weights.{epoch:02d}.hdf5') #{val_loss:.2f}
    #lambda epoch: 1e-3 * pow(0.75, floor(3 / 1))
    #schedule = lambda epoch, lr: list([lr]*5 + [lr/5]*10)[epoch]
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(ckpt_name, save_best_only=True), 
        #tf.keras.callbacks.LearningRateScheduler(schedule, verbose=1)
    ]
    
    if resume_training:
        ckpt_name = os.path.join(config['data_path'], config['experiment'], 'weights.01.hdf5')
        model.load_weights(ckpt_name)
        tf.keras.backend.set_value(model.optimizer.lr, 0.001/5)
            
    model.fit_generator(
        gen, 
        validation_data=val_gen,
        verbose=1, 
        use_multiprocessing=False, workers=config['workers'],
        #class_weight={0: 0.2, 1: 0.1, 2: 0.1, 3: 0.1, 4: 0.1, 5: 0.1},
        #steps_per_epoch=1000,
        epochs=3,
        callbacks=callbacks)
    

def evaluate(config):
    #test_df = util.read_testset()
    df = util.read_trainset()

    # train set (90%) and validation set (10%)
    from sklearn.model_selection import ShuffleSplit
    ss = ShuffleSplit(n_splits=5, test_size=0.1, random_state=42).split(df.index)

    # will just do one fold
    train_idx, valid_idx = next(ss)

    print(f"train: {len(train_idx)}, val: {len(valid_idx)}")
    
    val_gen = util.RsnaDataGenerator(
        df.iloc[valid_idx], 
        config['batch_size'], 
        config['image_size'], 
        augment_prob=None,
        #img_dir=config['train_path_dicom'],
        #img_ext='dcm'
        img_dir=config['train_path'],
        img_ext='jpg'
    )
        
    model = util.Model1(
        engine=ResNet50,        
        #engine=tf.keras.applications.InceptionResNetV2,
        input_dims=config['image_size'], batch_size=config['batch_size'],         
        weights="imagenet", 
        verbose=1).model
    
    ckpt_name = os.path.join(config['data_path'], config['experiment'], 'weights.01.hdf5')
    print(f'loading {ckpt_name}')
        
    model.load_weights(ckpt_name)
        
    preds = model.predict_generator(val_gen, verbose=1, workers=config['workers'])
    
    val_df = df.iloc[valid_idx]
    score = util.weighted_loss_metric(val_df['Label'].values, preds[:len(valid_idx)])
    return score

if __name__ == '__main__':
    #run(config, resume_training=True)
    score = evaluate(config)
    print(score)
