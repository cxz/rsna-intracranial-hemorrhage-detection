import sys

sys.path.insert(0, '.')

import os
import glob
import argparse
from datetime import datetime

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from sklearn.metrics import log_loss

import util

def check_val_pred(config, fold_no, epoch):    
    fold_path = util.fold_path(config, config['experiment'], fold_no)
    val_pred_fname = os.path.join(fold_path, f'val_preds.{epoch:02}.npy')    
    if not os.path.exists(val_pred_fname):
        return
    
    val_pred = np.load(val_pred_fname)    
    _, val_df = util.read_trainval(fold_no)    
    score = util.weighted_loss_metric(val_df[util.CLASSES].values, val_pred[:len(val_df)])
    
    #check log loss for each class
    per_class = []
    for c_idx, c in enumerate(util.CLASSES):
        per_class_loss = f'{log_loss(val_df[c].values, val_pred[:len(val_df), c_idx], eps=1e-7):.4f}'
        per_class.append(per_class_loss)
    
    print(config['experiment'], fold_no, epoch, f'{score:.4f}', per_class)
    
    
def check(config):
    for fold_no in [0, 1, 2, 3]:
        for epoch in [3, 4, 5]:
            check_val_pred(config, fold_no, epoch)
            
        
def predict(config):    
    df = util.read_testset()
    gen = util.RsnaDataGenerator(
        df, 
        config['batch_size'], 
        config['image_size'], 
        augment_prob=None,
        return_labels=False,
        img_dir=config['test_path_dicom'],
        img_ext='dcm'
    )
        
    model = util.Model1(
        engine=ResNet50,        
        input_dims=config['image_size'], batch_size=config['batch_size'],         
        weights="imagenet", 
        verbose=1).model
        
    epoch_folds = [
        [0, 4],
        [1, 4],
        [2, 3],
        [3, 5],
    ]
    
    for fold_no, epoch in epoch_folds:
        fold_path = util.fold_path(config, config['experiment'], fold_no)
        ckpt_name = os.path.join(fold_path, f'weights.{epoch:02}.hdf5')
        if not os.path.exists(ckpt_name):
            continue
                
        print(f'predicting test with ckpt {ckpt_name}')
        model.load_weights(ckpt_name)            
        preds = model.predict_generator(gen, verbose=1, workers=config['workers'])
        preds_name = os.path.join(fold_path, f'test_preds.fold{fold_no}-{epoch:02}.npy')
        np.save(preds_name, preds)
    
def submit(config):    
    experiment_path = os.path.join(config['data_path'], config['experiment'])
    fnames = list(glob.glob(os.path.join(experiment_path, '**', 'test_preds.fold*npy')))
    preds = np.zeros(shape=(78545, 6), dtype=np.float32)
    for fname in fnames:
        print(fname)
        fold_pred = np.load(fname)[:preds.shape[0]*6] #todo: *6 is temporary as workaround to bug in test_df
        print(fold_pred.shape)
        preds += fold_pred[::6]
    preds /= len(fnames)
    for c in range(6):
        print(c, np.mean(preds[:, c]))
        
    test_df = util.read_testset()
    
    submission_name = os.path.join(
        experiment_path, f'submission{datetime.now().strftime("%Y%m%d%H%M%S")}.csv')
    submission_df = pd.DataFrame()
    submission_df['ID'] = [f'{image_id}_{c}' for image_id in test_df.image.values for c in util.CLASSES]
    submission_df['Label'] = [preds[i, c] for i in range(preds.shape[0]) for c in range(6)]
    submission_df.to_csv(submission_name, index=False)

    print(submission_df.Label.max(), submission_df.Label.min())
        
    
if __name__ == '__main__':
    config = util.config
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', dest='experiment', default=config['experiment'])
    args = parser.parse_args()    
        
    config['experiment'] = args.experiment
    
    #check(config)
    #predict(config)
    submit(config)
    print('.')
    