import numpy as np
import pandas as pd
import pydicom
import os
import traceback
import matplotlib.pyplot as plt
import collections
from tqdm import tqdm_notebook as tqdm
from datetime import datetime
import json

from math import ceil, floor
import cv2

import warnings  
with warnings.catch_warnings():  
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf

#import keras

#import sys

#from keras_applications.resnet import ResNet50

import tensorflow as tf


from albumentations import (HorizontalFlip, CenterCrop, ElasticTransform, Blur, ShiftScaleRotate)
from albumentations import OneOf, Compose

def prepare_experiment(config):
    path = os.path.join(config['data_path'], config['experiment'])
    os.makedirs(path, exist_ok=True)
    
    with open(os.path.join(path, 'config.json'), 'w') as f:
        f.write(json.dumps(config))

    return path

def augmentation1(p):    
    return Compose([
        HorizontalFlip(),        
        #Blur(blur_limit=3),
        #OneOf([
        #    ElasticTransform(alpha=100, sigma=500, alpha_affine=10),
        #    ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=15),
        #], p=0.5)
    ], p=p)

def test_gpu():
    print(tf.__version__)
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.matmul(a, b)
    # Creates a session with log_device_placement set to True.
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    # Runs the op.
    print(sess.run(c))

def _get_first_of_dicom_field_as_int(x):
    if type(x) == pydicom.multival.MultiValue:
        return int(x[0])
    else:
        return int(x)

def _get_windowing(data):
    dicom_fields = [data.WindowCenter, data.WindowWidth, data.RescaleSlope, data.RescaleIntercept]
    return [_get_first_of_dicom_field_as_int(x) for x in dicom_fields]


def _window_image(img, window_center, window_width, slope, intercept):
    img = (img * slope + intercept)
    img_min = window_center - window_width//2
    img_max = window_center + window_width//2
    img[img<img_min] = img_min
    img[img>img_max] = img_max
    return img 

def _normalize(img):
    if img.max() == img.min():
        return np.zeros(img.shape)
    return 2 * (img - img.min())/(img.max() - img.min()) - 1


def _read_dicom(path, desired_size=(224, 224)):
    """Will be used in DataGenerator"""
    #print('reading ', path)
    dcm = pydicom.dcmread(path)

    window_center, window_width, slope, intercept = _get_windowing(dcm) 

    #(window_center, window_width)
    windows = [(30, 80), (80, 200), (600, 2800)]    
    img = np.zeros(shape=list(desired_size)+[len(windows)], dtype=np.float32)        
    for ch, w in enumerate(windows):
        try:             
            x = _window_image(dcm.pixel_array, w[0], w[1], slope, intercept)
            x = _normalize(x)
            if desired_size != (512, 512):
                x = cv2.resize(x, desired_size, interpolation=cv2.INTER_LINEAR)
        except:
            traceback.print_exc()
            x = np.zeros(desired_size)
            
        img[..., ch] = x
    
    return img

def _read_dicom3(path, desired_size=(224, 224)):
    try:
        dcm = pydicom.dcmread(path)        
        x = cv2.resize(dcm.pixel_array, desired_size, interpolation=cv2.INTER_LINEAR)   
        x = np.clip((x - 249.68)/888.93, -1.0, 1.0)
        return x[:,:,np.newaxis]
    except:
        return np.zeros(shape=desired_size)[:,:,np.newaxis]

def _read_dicom1(path, desired_size=(224, 224)):
    """Will be used in DataGenerator"""
    #print('reading ', path)
    dcm = pydicom.dcmread(path)

    window_params = _get_windowing(dcm) # (center, width, slope, intercept)

    try:
        # dcm.pixel_array might be corrupt (one case so far)
        img = _window_image(dcm.pixel_array, *window_params)
    except:
        img = np.zeros(desired_size)

    img = _normalize(img)

    if img.shape != desired_size:
        img = cv2.resize(img, desired_size, interpolation=cv2.INTER_LINEAR)

    return img[:,:,np.newaxis]

def _read_dicom2(path, desired_size=(224, 224)):
    dcm = pydicom.dcmread(path)        
    try:
        x = 2 * np.clip(dcm.pixel_array, 0, 2000)/2000.0 - 1
        if desired_size != dcm.pixel_array.shape :
            #print(path, ' pixel_array shape: ', dcm.pixel_array.shape)
            x = cv2.resize(x, desired_size, interpolation=cv2.INTER_LINEAR)
        return x[:, :, np.newaxis]
    except:
        traceback.print_exc()
        x = np.zeros(desired_size)        
        return x[:, :, np.newaxis]

def _read(path, desired_size=(224, 224)):
    try:
        img = cv2.imread(path)
        if desired_size != (512, 512):
            img = cv2.resize(img, desired_size, interpolation=cv2.INTER_LINEAR)
    except:
        print(path)
        raise
    return img[:, :, :1]

def calculating_class_weights(y_true):
    from sklearn.utils.class_weight import compute_class_weight
    number_dim = np.shape(y_true)[1]
    weights = np.empty([number_dim, 2])
    for i in range(number_dim):
        x = compute_class_weight('balanced', [0.,1.], y_true[:, i])        
        weights[i] = x
    return weights

# def get_weighted_loss(weights=np.array([
#        [  0.5, 0.5 ],
#        [  0.5,  20.],
#        [  0.5,  10.],
#        [  0.5,  15.],
#        [  0.5,  10.],
#        [  0.5,   5.]])
#                      ):
#     def weighted_loss(y_true, y_pred):
#         w = (weights[:,0]**(1-y_true))*(weights[:,1]**(y_true))
#         bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
#         return tf.keras.backend.mean(w*bce, axis=-1)
#     return weighted_loss

class RsnaDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, df, batch_size, img_size, 
                 augment_prob=None,
                 return_labels=True,
                 img_dir='../data/stage_1_test_images_jpg/', 
                 img_ext='jpg',
                 *args, **kwargs):
        print(f'building generator: {len(df)}, path: {img_dir}, augment: {augment_prob}')
        self.df = df
        self.batch_size = batch_size
        self.img_size = img_size
        self.img_dir = img_dir
        self.img_ext = img_ext
        self.augment = augmentation1(augment_prob) if augment_prob is not None else None
        self.return_labels = return_labels
        self.on_epoch_end()

    def __len__(self):
        return int(ceil(len(self.df) / self.batch_size))

    def __getitem__(self, index):
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        return self._data_generation(self.df.iloc[indices])

    def on_epoch_end(self):
        self.indices = np.arange(len(self.df))

    def _data_generation(self, batch_df):
        X = np.empty((self.batch_size, *self.img_size, 1))
        for i, ID in enumerate(batch_df.index.values):
            fname = os.path.join(self.img_dir, ID + '.' + self.img_ext)
            if self.img_ext == 'dcm':
                x = _read_dicom3(fname, self.img_size)
            else:
                x = _read(fname, self.img_size)
            if self.augment is not None:
                x = self.augment(image=x)['image']
            X[i,] = x
            
        if not self.return_labels:
            return X
                    
        Y = np.empty((self.batch_size, 6), dtype=np.float32)
        for i, label in enumerate(batch_df['Label'].values):
            Y[i,] = label
            
        return X, Y
    
    
class BalancedTrainDataGenerator(RsnaDataGenerator):
    def __init__(self, df, batch_size, img_size, 
                 augment_prob=None,
                 return_labels=True,                 
                 img_dir='../data/stage_1_train_images_jpg/', 
                 img_ext='jpg',
                 *args, **kwargs):
        super().__init__(df, batch_size, img_size, 
                         augment_prob=augment_prob,
                         return_labels=return_labels,
                         img_dir=img_dir, img_ext=img_ext) #, args, kwargs)        
        
    def __getitem__(self, index):
        #indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        #sample half with 'any'==1
        batch_pos = int(0.5 * self.batch_size)
        batch_neg = self.batch_size - batch_pos
        pos = self.df[self.df['Label']['any']==1].sample(n=batch_pos)
        neg = self.df[self.df['Label']['any']==0].sample(n=batch_neg)
        return self._data_generation(pd.concat([pos, neg]).sample(frac=1))


    



# def get_weighted_log_loss(y_true, y_pred):
    
#     # label "any" has twice the weight of the others
#     class_weights = np.array([1.8, 0.9, 0.9, 0.9, 0.9, 0.9]) 
    
#     eps = tf.keras.backend.epsilon()
#     y_pred = tf.keras.backend.clip(y_pred, eps, 1.0-eps)

#     out = -(y_true * tf.keras.backend.log(y_pred) * class_weights
#             + (1.0 - y_true) * tf.keras.backend.log(1.0 - y_pred) * class_weights)
    
#     return tf.keras.backend.mean(out, axis=-1)
                              

def _initial_layer(input_dims):
    inputs = tf.keras.layers.Input(input_dims)
    
    x = tf.keras.layers.Conv2D(
        filters=3, kernel_size=(1, 1), strides=(1, 1), name="initial_conv2d")(inputs)
    x = tf.keras.layers.BatchNormalization(axis=3, epsilon=1.001e-5, name='initial_bn')(x)
    x = tf.keras.layers.Activation('relu', name='initial_relu')(x)
    
    return tf.keras.models.Model(inputs, x)

class Model1:    
    def __init__(self, engine, input_dims, batch_size=5,
                 weights="imagenet", verbose=1): 
        self.engine = engine
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.weights = weights
        self.verbose = verbose
        self._build()

    def _build(self):
        
        initial_layer = _initial_layer((*self.input_dims, 1))
    
        engine = self.engine(
            include_top=False, 
            weights=self.weights, input_shape=(*self.input_dims, 3),
            backend = tf.keras.backend, layers = tf.keras.layers,
            models = tf.keras.models, utils = tf.keras.utils)

        x = engine(initial_layer.output)
        x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
        #x = tf.keras.layers.Dropout(0.5)(x)
        out = tf.keras.layers.Dense(6, activation="sigmoid", name='dense_output')(x)
        self.model = tf.keras.models.Model(inputs=initial_layer.input, outputs=out)
        
        #opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt)
        loss = 'binary_crossentropy'  #  get_weighted_loss()
        #loss = get_weighted_log_loss
        self.model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam())
        

def read_testset(filename="../input/stage_1_sample_submission.csv"):
    df = pd.read_csv(filename)
    df["Image"] = df["ID"].str.slice(stop=12)
    df["Diagnosis"] = df["ID"].str.slice(start=13)
    
    df = df.loc[:, ["Label", "Diagnosis", "Image"]]
    df = df.set_index(['Image', 'Diagnosis']).unstack(level=-1)
    
    return df

def read_trainset(filename="../input/stage_1_train.csv", sample=None):
    df = pd.read_csv(filename, nrows=sample)
    df["Image"] = df["ID"].str.slice(stop=12)
    df["Diagnosis"] = df["ID"].str.slice(start=13)

    df = df[~df.Image.isin(['ID_6431af929'])]
    
    duplicates_to_remove = [
        1598538, 1598539, 1598540, 1598541, 1598542, 1598543,
        312468,  312469,  312470,  312471,  312472,  312473,
        2708700, 2708701, 2708702, 2708703, 2708704, 2708705,
        3032994, 3032995, 3032996, 3032997, 3032998, 3032999
    ]
    
    if sample is None:
        df = df.drop(index=duplicates_to_remove)
        
    df = df.reset_index(drop=True)
    
    df = df.loc[:, ["Label", "Diagnosis", "Image"]]
    df = df.set_index(['Image', 'Diagnosis']).unstack(level=-1)
    
    return df


def weighted_loss_metric(trues, preds, weights=[0.2, 0.1, 0.1, 0.1, 0.1, 0.1], clip_value=1e-7):
    """this is probably not correct, but works OK. Feel free to give feedback."""
    preds = np.clip(preds, clip_value, 1-clip_value)
    loss_subtypes = trues * np.log(preds) + (1 - trues) * np.log(1 - preds)
    loss_weighted = np.average(loss_subtypes, axis=1, weights=weights)
    return - loss_weighted.mean()

def test_balanced_gen():
    df = read_trainset(sample=10000)
    gen = BalancedTrainDataGenerator(df, 64, (224, 224), return_labels=True)    
    import numpy as np
    for i in range(10):
        x, y = gen[i]
        print(x.shape, y.shape, np.sum(y, axis=0))

if __name__ == '__main__':
    test_balanced_gen()
