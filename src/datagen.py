import os
import tensorflow as tf
import numpy as np
import cv2
import random
from .utils import normalize


class DataGenerator:
    def __init__(self, content_data_path, 
                 style_data_path, 
                 smooth_style_data_path, 
                 batch_size=32,
                 image_size=(256,256),
                 prefetch=1):
        
        self.batch_size = batch_size
        self.image_size = image_size
        
        self.content_files = [os.path.join(content_data_path, fname) for fname in os.listdir(content_data_path)]
        self.style_files = [os.path.join(style_data_path, fname) for fname in os.listdir(style_data_path)]
        self.smooth_files = [os.path.join(smooth_style_data_path, fname) for fname in os.listdir(smooth_style_data_path)]
        self._balance_samples()
        
        self.content_ds = tf.data.Dataset\
                             .from_generator(
                                self._content_datagen(self.content_files), 
                                output_signature=tf.TensorSpec((*image_size,3)))
        
        self.style_ds = tf.data.Dataset\
                           .from_generator(
                                self._style_datagen(self.style_files, self.smooth_files), 
                                output_signature=(
                                    tf.TensorSpec((*image_size,3)), # style image
                                    tf.TensorSpec((*image_size,3)), # edge-smoothed style image
                                    tf.TensorSpec((*image_size,3)), # gray style image
                                ))
        
        self.dataset = tf.data.Dataset.zip((self.content_ds, self.style_ds))\
                                      .shuffle(300)\
                                      .batch(batch_size)\
                                      .prefetch(1)
        
        self.total_batches = int(np.ceil(len(self.content_files)/self.batch_size))
    
    def _balance_samples(self):
        '''
        Makes content data and style data have the same number of samples 
        or else tf.data will ignore surplus
        '''
        assert len(self.content_files) != 0
        assert len(self.style_files) != 0
        assert len(self.smooth_files) != 0
        assert len(self.style_files) == len(self.smooth_files), 'Mismatch between style data and edge-smoothed data'
        
        offset = len(self.content_files) - len(self.style_files)
        if offset > 0:
            extras = random.choices(self.style_files,k=offset)
            self.style_files+=extras
            self.smooth_files+=extras

        elif offset < 0:
            extras = random.choices(self.content_files,k=abs(offset))
            self.content_files+=extras
            
    def _content_datagen(self, files):
        def generator():
            for file in files:
                image = cv2.imread(file)
                image = cv2.resize(image, self.image_size)
                image = image[:,:,[2, 1 ,0]] # BGR -> RGB
                image = normalize(image)
                yield image
        return generator
    
    def _style_datagen(self, style_files, smooth_files):
        def generator():
            for f1, f2 in zip(style_files, smooth_files):
                style_image = cv2.imread(f1)
                style_image = cv2.resize(style_image, self.image_size)
                
                smooth_image = cv2.imread(f2)
                smooth_image = cv2.resize(smooth_image, self.image_size)
                
                gray = cv2.cvtColor(style_image, cv2.COLOR_BGR2GRAY)
                gray_image = np.stack([gray, gray, gray], axis=-1)
                
                style_image = style_image[:, :, [2, 1 ,0]] # BGR -> RGB
                smooth_image = smooth_image[:, :, [2, 1, 0]]
                
                style_image = normalize(style_image)
                smooth_image = normalize(smooth_image)
                gray_image = normalize(gray_image)
                
                yield style_image, smooth_image, gray_image
        return generator
    
    def __iter__(self):
        for samples in self.dataset:
            yield samples