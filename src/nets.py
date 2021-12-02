from .layers import BlockA, BlockB, BlockC, BlockD, BlockE, SN_ConvReflectionPad2D, DiscriminationLayer
import tensorflow as tf
k = tf.keras

class Generator(k.models.Model):
    def __init__(self, input_shape=(256,256,3), **kwargs):
        super().__init__(**kwargs)
        self.build_net(input_shape)
        
    def build_net(self, input_shape):
        original_size = input_shape[:-1]
        inp = k.layers.Input(input_shape)
        
        z = BlockA()(inp)
        
        half_size = z.shape[1:-1]
        
        z = BlockB()(z)
        z = BlockC()(z)
        z = BlockD(half_size)(z)      
        z = BlockE(original_size)(z)
        
        out = k.layers.Conv2D(filters=3, kernel_size=1, strides=1, use_bias=False, activation='tanh', name='output')(z)
        
        self.model = k.Model(inputs=inp, outputs=out, name='Generator')
        
    
    def call(self, x):
        out = self.model(x)
        return out
    
    def summary(self):
        return self.model.summary()
    
    
class Discriminator(k.models.Model):
    def __init__(self, input_shape=(256,256,3), init_channels=64, n_dis_layer=3, **kwargs):
        super().__init__(**kwargs)
        self.build_net(input_shape, init_channels, n_dis_layer)
        
    def build_net(self, input_shape, init_channels, n_dis_layer):
        channels = init_channels//2
        
        layers = [
            k.layers.InputLayer(input_shape),
            SN_ConvReflectionPad2D(filters=channels, kernel_size=3, strides=1),
            k.layers.LeakyReLU(alpha=0.2),
        ]
        
        for i in range(n_dis_layer):
            layers.append(DiscriminationLayer(channels))
            channels *= 2
        
        layers += [
            SN_ConvReflectionPad2D(filters=channels*2, kernel_size=3, strides=1),
            k.layers.LayerNormalization(epsilon=1e-5),
            k.layers.LeakyReLU(alpha=0.2),
            SN_ConvReflectionPad2D(filters=1, kernel_size=3, strides=1)
        ]
        
        self.model = k.Sequential(layers, name='Discriminator')
        
    
    def call(self, x):
        out = self.model(x)
        return out
    
    def summary(self):
        return self.model.summary()