import tensorflow as tf
import tensorflow_addons as tfa
k = tf.keras

class BaseLayer(k.layers.Layer):
    def summary(self):
        return self.model.summary()
    
    def __getitem__(self, key):
        return self.model.layers[key]


class RelfectionPad2D(k.layers.Layer):
    def __init__(self, pad_dims, **kwargs):
        super().__init__(**kwargs)
        self.pad_dims = pad_dims
    
    def call(self, x):
        out = tf.pad(x, paddings=[(0,0),*self.pad_dims,(0,0)], mode='REFLECT')
        return out

class ConvNormLReLu(BaseLayer):
    def __init__(self, pad_dims, filters, kernel_size, strides, use_bias=False, groups=1, **kwargs):
        super().__init__(**kwargs)
        self.model = k.Sequential([
            RelfectionPad2D(pad_dims),
            k.layers.Conv2D(filters, kernel_size, strides, use_bias=use_bias, groups=groups),
            k.layers.LayerNormalization(epsilon=1e-5),
            k.layers.LeakyReLU(alpha=0.2),
        ])
    
    def call(self, x):
        out = self.model(x)
        return out
    
class InvertedResBlock(BaseLayer):
    def __init__(self, in_filters, out_filters, expansion_ratio=2, **kwargs):
        super().__init__(**kwargs)
        
        self.use_res_connect = in_filters == out_filters
        
        bottleneck = int(round(in_filters * expansion_ratio))
        
        layers = []
        if expansion_ratio != 1:
            layers.append(ConvNormLReLu(pad_dims=[[0,0],[0,0]], 
                                        filters=bottleneck, 
                                        kernel_size=1, 
                                        strides=1))
            
        # DW
        layers.append(ConvNormLReLu(pad_dims=[[1,1],[1,1]], 
                                    filters=bottleneck, 
                                    kernel_size=3, 
                                    strides=1, 
                                    use_bias=True, 
                                    groups=bottleneck))
        
        # PW
        layers.append(k.layers.Conv2D(out_filters, kernel_size=1, strides=1))
        layers.append(k.layers.LayerNormalization(epsilon=1e-5))
        
        self.model = k.Sequential(layers)
        
    def call(self, x):
        out = self.model(x)
        if self.use_res_connect:
            out += x
        return out
    
class BlockA(BaseLayer):
    name = 'BlockA'
    def __init__(self, **kwargs):
        super().__init__(name=self.name, **kwargs)
        self.model = k.Sequential([
            ConvNormLReLu(pad_dims=[[3,3],[3,3]], filters=32, kernel_size=7, strides=1),
            ConvNormLReLu(pad_dims=[[0,1],[0,1]], filters=64, kernel_size=3, strides=2),
            ConvNormLReLu(pad_dims=[[1,1],[1,1]], filters=64, kernel_size=3, strides=1),
        ])
    
    def call(self, x):
        out = self.model(x)
        return out
    
class BlockB(BaseLayer):
    name = 'BlockB'
    def __init__(self, **kwargs):
        super().__init__(name=self.name, **kwargs)
        self.model = k.Sequential([
            ConvNormLReLu(pad_dims=[[0,1],[0,1]], filters=128, kernel_size=3, strides=2),
            ConvNormLReLu(pad_dims=[[1,1],[1,1]], filters=128, kernel_size=3, strides=1),
        ])
    
    def call(self, x):
        out = self.model(x)
        return out

class BlockC(BaseLayer):
    name = 'BlockC'
    def __init__(self, **kwargs):
        super().__init__(name=self.name, **kwargs)
        self.model = k.Sequential([
            ConvNormLReLu(pad_dims=[[1,1],[1,1]], filters=128, kernel_size=3, strides=1),
            InvertedResBlock(in_filters=128, out_filters=256),
            InvertedResBlock(in_filters=256, out_filters=256),
            InvertedResBlock(in_filters=256, out_filters=256),
            InvertedResBlock(in_filters=256, out_filters=256),
            ConvNormLReLu(pad_dims=[[1,1],[1,1]], filters=128, kernel_size=3, strides=1)
        ])
        
    def call(self, x):
        out = self.model(x)
        return out
    
class BlockD(BaseLayer):
    name = 'BlockD'
    def __init__(self, upsample_size, **kwargs):
        super().__init__(name=self.name, **kwargs)
        
        self.upsample_size = upsample_size
        self.model = k.Sequential([
            ConvNormLReLu(pad_dims=[[1,1],[1,1]], filters=128, kernel_size=3, strides=1),
            ConvNormLReLu(pad_dims=[[1,1],[1,1]], filters=128, kernel_size=3, strides=1),
        ])
    
    def call(self, x):
        x = tf.image.resize(x, size=self.upsample_size, antialias=True)
        out = self.model(x)
        return out
    
class BlockE(BaseLayer):
    name = 'BlockE'
    def __init__(self, upsample_size, **kwargs):
        super().__init__(name=self.name, **kwargs)
        
        self.upsample_size = upsample_size
        self.model = k.Sequential([
            ConvNormLReLu(pad_dims=[[1,1],[1,1]], filters=64, kernel_size=3, strides=1),
            ConvNormLReLu(pad_dims=[[1,1],[1,1]], filters=64, kernel_size=3, strides=1),
            ConvNormLReLu(pad_dims=[[3,3],[3,3]], filters=32, kernel_size=7, strides=1),
        ])
    
    def call(self, x):
        x = tf.image.resize(x, size=self.upsample_size, antialias=True)
        out = self.model(x)
        return out
    
    
class SN_ConvReflectionPad2D(BaseLayer):
    def __init__(self, filters, kernel_size, strides, pad=1, use_bias=False, activation=None, **kwargs):
        super().__init__(**kwargs)
        if (kernel_size - strides) % 2 == 0 :
            pad_top = pad
            pad_bottom = pad
            pad_left = pad
            pad_right = pad

        else :
            pad_top = pad
            pad_bottom = kernel_size - strides - pad_top
            pad_left = pad
            pad_right = kernel_size - strides - pad_left
            
        self.model = k.Sequential([
            RelfectionPad2D([[pad_top,pad_bottom],[pad_left,pad_right]]),
            tfa.layers.SpectralNormalization(
                k.layers.Conv2D(filters=filters, 
                                kernel_size=kernel_size, 
                                strides=strides, 
                                use_bias=use_bias,
                                activation=activation))
        ])
    
    def call(self, x):
        out = self.model(x)
        return out
    

class DiscriminationLayer(BaseLayer):
    def __init__(self, channels, **kwargs):
        super().__init__(**kwargs)
        self.model = k.Sequential([
            SN_ConvReflectionPad2D(filters=channels*2, kernel_size=3, strides=2),
            k.layers.LeakyReLU(alpha=0.2),
            SN_ConvReflectionPad2D(filters=channels*4, kernel_size=3, strides=1),
            k.layers.LayerNormalization(epsilon=1e-5),
            k.layers.LeakyReLU(alpha=0.2),
        ])
        
    def call(self, x):
        out = self.model(x)
        return out