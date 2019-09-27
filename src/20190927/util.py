import numpy as np
import pandas as pd
import pydicom
import os
import matplotlib.pyplot as plt
import collections
from tqdm import tqdm_notebook as tqdm
from datetime import datetime

from math import ceil, floor
import cv2

import tensorflow as tf
#import keras

#import sys

#from keras_applications.resnet import ResNet50

import tensorflow as tf




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

    window_params = _get_windowing(dcm) # (center, width, slope, intercept)

    try:
        # dcm.pixel_array might be corrupt (one case so far)
        img = _window_image(dcm.pixel_array, *window_params)
    except:
        img = np.zeros(desired_size)

    img = _normalize(img)

    if desired_size != (512, 512):
        # resize image
        img = cv2.resize(img, desired_size, interpolation=cv2.INTER_LINEAR)

    return img[:,:,np.newaxis]

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
                 return_labels=True,
                 img_dir='../data/stage_1_test_images_jpg/', *args, **kwargs):
        self.df = df
        self.batch_size = batch_size
        self.img_size = img_size
        self.img_dir = img_dir
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
            X[i,] = _read(self.img_dir+ID+".jpg", self.img_size)
            
        if not self.return_labels:
            return X
                    
        Y = np.empty((self.batch_size, 6), dtype=np.float32)
        for i, label in enumerate(batch_df['Label'].values):
            Y[i,] = label
            
        return X, Y
    
    
class BalancedTrainDataGenerator(RsnaDataGenerator):
    def __init__(self, df, batch_size, img_size, return_labels=True,
                 img_dir='../data/stage_1_train_images_jpg/', *args, **kwargs):
        super().__init__(df, batch_size, img_size, return_labels, img_dir, args, kwargs)
        print('building balanced train generator: ', len(df), img_dir)
        
    def __getitem__(self, index):
        #indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        #sample half with 'any'==1
        pos = self.df[self.df['Label']['any']==1].sample(n=self.batch_size//2)
        neg = self.df[self.df['Label']['any']==0].sample(n=self.batch_size//2)
        return self._data_generation(pd.concat([pos, neg]).sample(frac=1))

    
class TrainDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, list_IDs, labels, batch_size=1, img_size=(512, 512), 
                 img_dir='../data/stage_1_train_images_jpg/', *args, **kwargs):
        print('building train generator: ', len(list_IDs), img_dir)
        self.list_IDs = list_IDs
        self.labels = labels
        self.batch_size = batch_size
        self.img_size = img_size
        self.img_dir = img_dir
        self.on_epoch_end()

    def __len__(self):
        return int(ceil(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indices]
        X, Y = self.__data_generation(list_IDs_temp)
        return X, Y

    def on_epoch_end(self):
        self.indices = np.arange(len(self.list_IDs))
        np.random.shuffle(self.indices)

    def __data_generation(self, list_IDs_temp):        
        X = np.empty((self.batch_size, *self.img_size, 1))
        Y = np.empty((self.batch_size, 6), dtype=np.float32)        
        for i, ID in enumerate(list_IDs_temp):
            X[i,] = _read(self.img_dir+ID+".jpg", self.img_size)
            Y[i,] = self.labels.loc[ID].values        
        return X, Y
    
    
class TestDataGenerator(tf.keras.utils.Sequence):

    def __init__(self, list_IDs, labels, batch_size=1, img_size=(512, 512), 
                 img_dir='../data/stage_1_test_images_jpg/', *args, **kwargs):

        self.list_IDs = list_IDs
        self.labels = labels
        self.batch_size = batch_size
        self.img_size = img_size
        self.img_dir = img_dir
        self.on_epoch_end()

    def __len__(self):
        return int(ceil(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indices]
        X = self.__data_generation(list_IDs_temp)
        return X

    def on_epoch_end(self):
        self.indices = np.arange(len(self.list_IDs))

    def __data_generation(self, list_IDs_temp):
        X = np.empty((self.batch_size, *self.img_size, 1))
        
        for i, ID in enumerate(list_IDs_temp):
            X[i,] = _read(self.img_dir+ID+".jpg", self.img_size)
        
        return X




def _initial_layer(input_dims):
    inputs = tf.keras.layers.Input(input_dims)
    
    x = tf.keras.layers.Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), name="initial_conv2d")(inputs)
    x = tf.keras.layers.BatchNormalization(axis=3, epsilon=1.001e-5, name='initial_bn')(x)
    x = tf.keras.layers.Activation('relu', name='initial_relu')(x)
    
    return tf.keras.models.Model(inputs, x)

class Model1:    
    def __init__(self, engine, input_dims, batch_size=5, learning_rate=1e-3, 
                 decay_rate=1.0, decay_steps=1, weights="imagenet", verbose=1):        
        self.engine = engine
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
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
        out = tf.keras.layers.Dense(6, activation="sigmoid", name='dense_output')(x)
        loss = 'binary_crossentropy'  #  get_weighted_loss()
        self.model = tf.keras.models.Model(inputs=initial_layer.input, outputs=out)
        self.model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(0.0))
        
    
    
    def fit(self, df, train_idx, img_dir, global_epoch):
        train_gen = BalancedTrainDataGenerator(
                df.iloc[train_idx].index, 
                df.iloc[train_idx], 
                self.batch_size, 
                self.input_dims, 
                img_dir)
        self.model.fit_generator(
            train_gen,
            #class_weight={0: 1, 1: 1, 2:, 1, 3: 1, 4: 1, 5: 1},
            verbose=self.verbose,
            use_multiprocessing=False,
            workers=4,
            callbacks=[
                tf.keras.callbacks.ModelCheckpoint('../data/weights.{epoch:02d}-{val_loss:.2f}.hdf5'),                
                tf.keras.callbacks.LearningRateScheduler(
                    lambda epoch: self.learning_rate * pow(self.decay_rate, floor(global_epoch / self.decay_steps))
                )
            ]
        )
    
    def predict(self, df, test_idx, img_dir):
        test_gen = TestDataGenerator(
                df.iloc[test_idx].index, 
                None, 
                self.batch_size, 
                self.input_dims, 
                img_dir)
        predictions = self.model.predict_generator(
            test_gen,
            verbose=1,
            use_multiprocessing=False,
            workers=4)
        return predictions[:df.iloc[test_idx].shape[0]]
    
    def save(self, path):
        self.model.save_weights(path)
    
    def load(self, path):
        self.model.load_weights(path)

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



def run(model, df, train_idx, valid_idx, test_df, epochs):
    
    valid_predictions = []
    test_predictions = []
    for global_epoch in range(epochs):

        model.fit(df, train_idx, _TRAIN_IMAGES, global_epoch)
        
        test_predictions.append(model.predict(test_df, range(test_df.shape[0]), _TEST_IMAGES))
        valid_predictions.append(model.predict(df, valid_idx, _TRAIN_IMAGES))
        
        print("validation loss: %.4f" %
              weighted_loss_metric(df.iloc[valid_idx].values, 
                                   np.average(valid_predictions, axis=0, 
                                              weights=[2**i for i in range(len(valid_predictions))]))
             )
    
    return test_predictions, valid_predictions



def weighted_loss_metric(trues, preds, weights=[0.2, 0.1, 0.1, 0.1, 0.1, 0.1], clip_value=1e-7):
    """this is probably not correct, but works OK. Feel free to give feedback."""
    preds = np.clip(preds, clip_value, 1-clip_value)
    loss_subtypes = trues * np.log(preds) + (1 - trues) * np.log(1 - preds)
    loss_weighted = np.average(loss_subtypes, axis=1, weights=weights)
    return - loss_weighted.mean()

def prepare_submission():
    test_df.iloc[:, :] = np.average(test_preds, axis=0, weights=[2**i for i in range(len(test_preds))])
    test_df = test_df.stack().reset_index()
    test_df.insert(loc=0, column='ID', value=test_df['Image'].astype(str) + "_" + test_df['Diagnosis'])
    test_df = test_df.drop(["Image", "Diagnosis"], axis=1)
    test_df.to_csv('../data/submission.csv', index=False)
    
    
def test_balanced_gen():
    df = read_trainset(sample=10000)
    gen = BalancedTrainDataGenerator(df, 64, (224, 224), return_labels=True)    
    import numpy as np
    for i in range(10):
        x, y = gen[i]
        print(x.shape, y.shape, np.sum(y, axis=0))
    
def build_generator(kind='validation', return_labels=None):
    batch_size = 64
    input_dims = (224, 224)
    img_dir = _TRAIN_IMAGES
    gen = util.RsnaDataGenerator(
            df.iloc[train_idx],  
            batch_size=batch_size, 
            img_size=input_dims, 
            img_dir=img_dir,
            return_labels=return_labels)
    return gen

if __name__ == '__main__':
    test_balanced_gen()