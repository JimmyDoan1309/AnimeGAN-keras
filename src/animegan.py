import tensorflow as tf
import os
from .nets import Generator, Discriminator
from .losses import *

VGG_OUT = 10

class Hyperparams:
    def __init__(self, gan_type='lsgan', 
                 init_lr=2e-4, 
                 gen_lr=2e-5, 
                 dis_lr=4e-5, 
                 gp_lambda=10.0, 
                 gen_adv_weight=300.0, 
                 dis_adv_weight=300.0, 
                 content_weight=1.2, 
                 style_weight=2.0, 
                 color_weight=10.0, 
                 tv_weight=1.0,
                 dis_loss_weight={
                     'style_w': 1.7, 
                     'fake_w': 1.7, 
                     'gray_w': 1.7,
                     'smooth_w': 1.0,
                 }):
        
        self.gan_type = gan_type
        self.init_lr = init_lr
        self.gen_lr = gen_lr
        self.dis_lr = dis_lr
        self.gp_lambda = gp_lambda
        self.gen_adv_weight = gen_adv_weight
        self.dis_adv_weight = dis_adv_weight
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.color_weight = color_weight
        self.dis_loss_weight = dis_loss_weight
        
    @classmethod
    def hayao(cls):
        return cls(content_weight=1.5, 
                   style_weight=2.5, 
                   color_weight=15.0, 
                   tv_weight=1.0,
                   dis_loss_weight={
                       'style_w': 1.2, 
                       'fake_w': 1.2,
                       'gray_w': 1.2,
                       'smooth_w': 0.8,
                   })
    
    @classmethod
    def paprika(cls):
        return cls(content_weight=2.0, 
                   style_weight=0.6, 
                   color_weight=50.0, 
                   tv_weight=0.1,
                   dis_loss_weight={
                       'style_w': 1.0, 
                       'fake_w': 1.0,
                       'gray_w': 1.0,
                       'smooth_w': 0.005,
                   })
    
    @classmethod
    def shinkai(cls):
        return cls(content_weight=1.2, 
                   style_weight=2.0, 
                   color_weight=10.0, 
                   tv_weight=1.0,
                   dis_loss_weight={
                       'style_w': 1.7, 
                       'fake_w': 1.7,
                       'gray_w': 1.7,
                       'smooth_w': 1.0,
                   })
    
    

class AnimeGAN:
    def __init__(self, input_shape, params, **kwargs):
        super().__init__(**kwargs)
        
        self.config = params
        
        self.gen = Generator(input_shape)
        self.dis = Discriminator(input_shape)
        vgg = tf.keras.applications.VGG19(include_top=False, input_shape=input_shape)
        self.vgg = tf.keras.Model(inputs=vgg.inputs, outputs=vgg.layers[VGG_OUT].output)
        
        self.init_optimizer = tf.optimizers.Adam(self.config.init_lr, beta_1=0.5)
        self.gen_optimizer = tf.optimizers.Adam(self.config.init_lr, beta_1=0.5)
        self.dis_optimizer = tf.optimizers.Adam(self.config.init_lr, beta_1=0.5)
        
        self.saved_freq = kwargs.get('saved_freq', 5)
        self.saved_path = kwargs.get('saved_path', './saved_model/')
        self.models_save = kwargs.get('saved_models', 'all')
        
        assert self.models_save in ['all', 'generator']
        
    def train(self, dataset, total_epochs=100, init_epochs=10):
        pass
    
    def _init_train_step(self, content):
        with tf.GradientTape() as tape:
            reconstruct = self.gen(content)
            loss = self.config.content_weight * content_loss(content, reconstruct, self.vgg)
        
        grads = tape.gradient(loss, self.gen.trainable_weights)
        self.init_optimizer.apply_gradients(zip(grads, self.gen.trainable_weights))
        return loss
    
    def _init_train(self, dataset, epochs):
        print('Init Training')
        for e in range(1, epochs+1):
            print('Epoch', e)
            for b, (content, (style, smooth, gray)) in enumerate(dataset, start=1):
                loss = self._init_train_step(content)
                print(f'\rBatch {b}/{dataset.total_batches}: reconstruct_loss = {loss}', end='', flush=True)
            print()
    
    def _train_step(self, content, style, smooth, gray):
        pass
    
    def _train(self, dataset, epochs):
        print('Adverserial Training')
        for e in range(1, epochs+1):
            print('Epoch', e)
            for b, (content, (style, smooth, gray)) in enumerate(dataset, start=1):
                gloss, dloss = self._train_step(content, style, smooth, gray)
                print(f'\rBatch {b}/{dataset.total_batches}: gloss = {gloss}, dloss = {dloss}', end='', flush=True)
            print()
            
            # saved model
            if e % self.saved_freq == 0 or e == epochs:
                self.save_models()
    
    def save_models(self):
        pass