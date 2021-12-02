import tensorflow as tf
import os
import json
import cv2
from .nets import Generator, Discriminator
from .losses import *
from .utils import denormalize


VGG_OUT = 10

class AnimeGAN:
    def __init__(self, input_shape, params):
        
        self.input_shape = input_shape
        self.params = params
        
        self.gen = Generator(input_shape)
        self.dis = Discriminator(input_shape)
        vgg = tf.keras.applications.VGG19(include_top=False, input_shape=input_shape)
        self.vgg = tf.keras.Model(inputs=vgg.inputs, outputs=vgg.layers[VGG_OUT].output)
        
        self.init_optimizer = tf.optimizers.Adam(self.params.init_lr, beta_1=0.5)
        self.gen_optimizer = tf.optimizers.Adam(self.params.gen_lr, beta_1=0.5)
        self.dis_optimizer = tf.optimizers.Adam(self.params.dis_lr, beta_1=0.5)
        
    def train(self, 
              dataset, 
              total_epochs=100, 
              init_epochs=10, 
              save_freq=1, 
              save_path='./model', 
              save_discriminator=True, 
              test_image=None, 
              test_generated_save_path=None):
        
        if test_image:
            assert test_generate_save_path is not None, 'Must provide test_generated_save_path'
            os.makedirs(test_generate_save_path, exist_ok=True)
            
            if len(test_image.shape) == 3:
                test_image = tf.expand_dims(test_image, axis=0)
        
        self._init_train(dataset, init_epochs, save_freq, save_path)
        self._train(dataset, 
                    total_epochs - init_epochs, 
                    save_freq, 
                    save_path, 
                    save_discriminator, 
                    test_image, 
                    test_generate_save_path)
        
    
    def _init_train(self, dataset, epochs, save_freq, save_path):
        print('Init Training')
        for e in range(1, epochs+1):
            print('Epoch', e)
            for b, (content, (style, smooth, gray)) in enumerate(dataset, start=1):
                loss = self._init_train_step(content)
                print(f'\rBatch {b}/{dataset.total_batches}: reconstruct_loss = {loss}', end='', flush=True)
            print()
            
            # saved model
            if e % saved_freq == 0 or e == epochs:
                self.save_models(save_path, False, verbose=0)
    
    def _train(self, dataset, epochs, save_freq, save_path, save_discriminator, test_image, test_generate_save_path):
        print('Adverserial Training')
        for e in range(1, epochs+1):
            print('Epoch', e)
            for b, (content, (style, smooth, gray)) in enumerate(dataset, start=1):
                gloss, dloss = self._train_step(content, style, smooth, gray)
                print(f'\rBatch {b}/{dataset.total_batches}: gloss = {gloss}, dloss = {dloss}', end='', flush=True)
            print()
            
            # saved model
            if e % saved_freq == 0 or e == epochs:
                self.save_models(save_path, save_discriminator, verbose=0)
            
            if test_image:
                self._test_generator(test_image, f'image_e_{e:04d}.jpeg', test_generate_save_path)
                
    def _init_train_step(self, content):
        with tf.GradientTape() as tape:
            reconstruct = self.gen(content)
            loss = self.params.content_weight * content_loss(content, reconstruct, self.vgg)
        
        grads = tape.gradient(loss, self.gen.trainable_weights)
        self.init_optimizer.apply_gradients(zip(grads, self.gen.trainable_weights))
        return loss.numpy()
    
    def _train_step(self, content, style, smooth, gray):
        with tf.GradientTape(persistent=True) as tape:
            generate = self.gen(content)
            generate_logit = self.dis(generate)
            style_logit = self.dis(style)
            smooth_logit = self.dis(smooth)
            gray_logit = self.dis(gray)
            
            content_loss, style_loss = content_style_loss(content, generate, style, self.vgg)
            tv_loss = total_variance_loss(generate)
            col_loss = color_loss(content, generate)
            
            gen_loss_1 = self.params.content_weight * content_loss + \
                         self.params.style_weight * style_loss + \
                         self.params.tv_weight * tv_loss + \
                         self.params.color_weight * col_loss
            
            gen_loss_2 = self.params.gen_adv_weight * generator_loss(generate_logit, self.params.gan_type)
            
            gen_loss = gen_loss_1 + gen_loss_2
            
            gp = gradient_penalty(style, generate, self.dis, self.params.gan_type, self.params.gp_lambda)
            
            dis_loss = gp + self.params.dis_adv_weight * discriminator_loss(style_logit, 
                                                                            smooth_logit, 
                                                                            gray_logit, 
                                                                            generate_logit,
                                                                            self.params.dis_loss_weight, 
                                                                            self.params.gan_type)
            
        gen_grads = tape.gradient(gen_loss, self.gen.trainable_weights)
        dis_grads = tape.gradient(dis_loss, self.dis.trainable_weights)
        
        del tape
        
        self.gen_optimizer.apply_gradients(zip(gen_grads, self.gen.trainable_weights))
        self.dis_optimizer.apply_gradients(zip(dis_grads, self.dis.trainable_weights))
            
        return gen_loss.numpy(), dis_loss.numpy()
    
    def _test_generator(self, image, file_name, save_path):
        generate = self.gen(image)
        generate = denormalize(generate[0].numpy(), as_float=False)
        cv2.imwrite(os.path.join(save_path, file_name), generate)
        
    
    def save(self, save_path, save_discriminator, verbose=1):
        os.makedirs(save_path, exist_ok=True)
        self.gen.save_weights(os.path.join(save_path,'generator/generator'))
        
        if save_discriminator:
            self.dis.save_weights(os.path.join(save_path, 'discriminator/discriminator'))
        
        config = {
            'input_shape': self.input_shape,
            'hyperparams': self.params.to_dict()
        }
        
        with open(os.path.join(save_path, 'config.json'), 'w+') as fp:
            json.dump(config, fp)
        
        if verbose:
            print(f'Model save at {save_path}')
    
    
    @classmethod
    def load(cls, path):
        assert os.path.exists(os.path.join(path, 'config.json')), 'No config.json file found'
        assert os.path.exists(os.path.join(path, 'generator')), 'No checkpoint for generator model found'
        
        with open(os.path.join(path, 'config.json'), 'r') as fp:
            config = json.load(fp)
        
        input_shape = config['input_shape']
        params = Hyperparams(**config['hyperparams'])
        
        gan = cls(input_shape, params)
        
        print('Load generator model')
        gan.gen.load_weights(os.path.join(path, 'generator/generator'))
        if os.path.exists(os.path.join(path, 'discriminator')):
            print('Load discriminator model')
            gan.dis.load_weights(os.path.join(path, 'discriminator/discriminator'))
        
        return gan
        
    
    
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
        self.tv_weight = tv_weight
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
    

    def to_dict(self):
        return {
            'gan_type': self.gan_type,
            'init_lr': self.init_lr,
            'gen_lr': self.gen_lr,
            'dis_lr': self.dis_lr,
            'gp_lambda': self.gp_lambda,
            'gen_adv_weight': self.gen_adv_weight,
            'dis_adv_weight': self.dis_adv_weight,
            'content_weight': self.content_weight,
            'style_weight': self.style_weight,
            'color_weight': self.color_weight,
            'tv_weight': self.tv_weight,
            'dis_loss_weight': self.dis_loss_weight,
        }