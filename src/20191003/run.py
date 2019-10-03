import sys

sys.path.insert(0, '.')

import os
import glob
import argparse

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50

import util

#notes: multi label weighting, fixed folds, load from dcms

def run(config, resume_training=False):
        
    experiment_path = util.prepare_experiment(config)
    fold_no = str(config['fold_no'])
    print(f"running {experiment_path}, fold_no={fold_no}")
    
    train_df, val_df = util.read_trainval(fold_no)
    batch_size = config['batch_size']
    print(f"train: {len(train_df)}, val: {len(val_df)}")
    print(f"train_steps: {len(train_df)//batch_size}")

    gen = util.BalancedTrainDataGenerator(
        train_df, 
        config['batch_size'], 
        config['image_size'], 
        augment_prob=config['augment_prob'], 
        img_dir=config['train_path_dicom'], 
        img_ext='dcm'
        #img_dir=config['train_path'], 
        #img_ext='jpg'
    )
    
    val_gen = util.RsnaDataGenerator(
        val_df, 
        config['batch_size'], 
        config['image_size'], 
        augment_prob=None,
        img_dir=config['train_path_dicom'],
        img_ext='dcm'
        #img_dir=config['train_path'],
        #img_ext='jpg'
    )
    
    model = util.Model1(
        engine=ResNet50,        
        #engine=tf.keras.applications.InceptionResNetV2,
        input_dims=config['image_size'], batch_size=config['batch_size'],         
        weights="imagenet", 
        verbose=1).model

    ckpt_name = os.path.join(experiment_path, 'weights.{epoch:02d}.hdf5') #{val_loss:.2f}
    #lambda epoch: 1e-3 * pow(0.75, floor(3 / 1))
    schedule = lambda epoch, lr: list([1e-3, 5e-4, 2e-4, 2e-4] + [1e-4]*10)[epoch]
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(ckpt_name), 
        tf.keras.callbacks.LearningRateScheduler(schedule, verbose=1)
    ]
    
    if resume_training:
        ckpt_name = os.path.join(config['data_path'], config['experiment'], 'weights.01.hdf5')
        model.load_weights(ckpt_name)
        tf.keras.backend.set_value(model.optimizer.lr, 0.001/5)
            
    model.fit_generator(
        gen, 
        #validation_data=val_gen,
        verbose=1, 
        use_multiprocessing=False, workers=config['workers'],
        #class_weight={0: 0.2, 1: 0.1, 2: 0.1, 3: 0.1, 4: 0.1, 5: 0.1},
        #steps_per_epoch=1000,
        epochs=5,
        callbacks=callbacks)
    

def evaluate(config):
    fold_no = str(config['fold_no'])
    _, val_df = util.read_trainval(fold_no)
    
    val_gen = util.RsnaDataGenerator(
        val_df, 
        config['batch_size'], 
        config['image_size'], 
        augment_prob=None,
        img_dir=config['train_path_dicom'],
        img_ext='dcm'
        #img_dir=config['train_path'],
        #img_ext='jpg'
    )
        
    model = util.Model1(
        engine=ResNet50,        
        input_dims=config['image_size'], batch_size=config['batch_size'],         
        weights="imagenet", 
        verbose=1).model
    
    fold_path = util.prepare_experiment(config, makedirs=False)
    ckpt_dir = os.path.join(fold_path, '*.hdf5')    
    scores = {}
    
    for epoch in [3, 4, 5]:
        ckpt_name = os.path.join(fold_path, f'weights.{epoch:02}.hdf5')
        if not os.path.exists(ckpt_name):
            continue
            
        print(f'loading {ckpt_name}')        
        model.load_weights(ckpt_name)        
        preds = model.predict_generator(val_gen, verbose=1, workers=config['workers'])    
        score = util.weighted_loss_metric(val_df[util.CLASSES].values, preds[:len(val_df)])
        scores[ckpt_name] = score
        print(f'{ckpt_name}: {score}')
        
        preds_name = os.path.join(fold_path, f'val_preds.{epoch:02}.npy')
        np.save(preds_name, preds)
        
    return scores


if __name__ == '__main__':
    config = util.config
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', dest='fold', default=config['fold_no'])
    parser.add_argument('--experiment', dest='experiment', default=config['experiment'])
    args = parser.parse_args()    
    
    config['fold_no'] = args.fold
    config['experiment'] = args.experiment
    
    #run(config, resume_training=False)
    scores = evaluate(config)
    print(scores)    

