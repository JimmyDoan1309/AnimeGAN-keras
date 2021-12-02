import tensorflow as tf
from .utils import denormalize

def gradient_penalty(style, fake, discriminator, gan_type='lsgan', gp_lambda=10.0):
    if gan_type in ['lsgan', 'gan', 'hinge']:
        return 0
    
    if gan_type == 'dragan':
        eps = tf.random.uniform(shape=tf.shape(style), minval=0., maxval=1.)
        _, x_var = tf.nn.moments(style, axes=[0, 1, 2, 3])
        x_std = tf.sqrt(x_var)  # magnitude of noise decides the size of local region

        fake = style + 0.5 * x_std * eps
    
    batch_size = style.shape[0]
    alpha = tf.random.uniform(shape=[batch_size, 1, 1, 1], minval=0., maxval=1.)
    interpolated = style + alpha * (fake - style)
    
    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        out = dis(interpolated)
    
    grad = tape.gradient(out, interpolated)
    grad_norm = tf.norm(tf.reshape(grad, [1,-1]), axis=1) # l2 norm
    
    gp = 0
    if gan_type == 'wgan-lp':
        gp = gp_lambda * tf.reduce_mean(tf.square(tf.maximum(0.0, grad_norm - 1.)))

    elif gan_type in ['wgan-gp', 'dragan']:
        gp = gp_lambda * tf.reduce_mean(tf.square(grad_norm - 1.))
        
    return gp

def gram(x):
    b = x.shape[0]
    c = x.shape[3]
    x = tf.reshape(x, [b,-1,c])
    gram = tf.matmul(x,x, transpose_a=True) / tf.cast(tf.size(x)//b, tf.float32)
    return gram

def content_loss(real, fake, vgg):
    real_feat = vgg(real)
    fake_feat = vgg(fake)
    loss = tf.losses.mae(real_feat, fake_feat)
    return tf.reduce_mean(loss)

def style_loss(style, fake, vgg):
    style_feat = vgg(style)
    fake_feat = vgg(fake)
    loss = tf.losses.mae(gram(style_feat), gram(fake_feat))
    return tf.reduce_mean(loss)

def content_style_loss(real, fake, style, vgg):
    c_loss = content_loss(real, fake, vgg)
    s_loss = style_loss(style, fake, vgg)
    
    return tf.reduce_mean(c_loss), tf.reduce_mean(s_loss)

def color_loss(real, fake):
    real = denormalize(real)
    fake = denormalize(fake)
    real = tf.image.rgb_to_yuv(real)
    fake = tf.image.rgb_to_yuv(fake)
    
    loss = tf.losses.mae(real[...,0], fake[...,0]) + \
            tf.losses.huber(real[...,1], fake[...,1]) + \
            tf.losses.huber(real[...,2], fake[...,2])
    return tf.reduce_mean(loss)

def total_variance_loss(inputs):
    dh = inputs[:, :-1, ...] - inputs[:, 1:, ...]
    dw = inputs[:, :, :-1, ...] - inputs[:, :, 1:, ...]
    size_dh = tf.size(dh, out_type=tf.float32)
    size_dw = tf.size(dw, out_type=tf.float32)
    return tf.nn.l2_loss(dh) / size_dh + tf.nn.l2_loss(dw) / size_dw

def discriminator_loss(style, smooth, gray, fake, gan_type='lsgan', **kwargs):
    style_w = kwargs.get('style_w', 1.7)
    fake_w = kwargs.get('fake_w', 1.7)
    gray_w = kwargs.get('gray_w', 1.7)
    smooth_w = kwargs.get('smooth_w', 1.0)
    
    style_loss = 0
    gray_loss = 0
    fake_loss = 0
    smooth_loss = 0

    if gan_type == 'wgan-gp' or gan_type == 'wgan-lp':
        style_loss = -tf.reduce_mean(style)
        gray_loss = tf.reduce_mean(gray)
        fake_loss = tf.reduce_mean(fake)
        smooth_loss = tf.reduce_mean(smooth)

    if gan_type == 'lsgan' :
        style_loss = tf.reduce_mean(tf.square(style - 1.0))
        gray_loss = tf.reduce_mean(tf.square(gray))
        fake_loss = tf.reduce_mean(tf.square(fake))
        smooth_loss = tf.reduce_mean(tf.square(smooth))

    if gan_type == 'gan' or gan_type == 'dragan' :
        style_loss = tf.reduce_mean(tf.losses.binary_crossentropy(tf.ones_like(style), style))
        gray_loss = tf.reduce_mean(tf.losses.binary_crossentropy(tf.zeros_like(gray), gray))
        fake_loss = tf.reduce_mean(tf.losses.binary_crossentropy(tf.zeros_like(fake), fake))
        smooth_loss = tf.reduce_mean(tf.losses.binary_crossentropy(tf.zeros_like(smooth), smooth))

    if gan_type == 'hinge':
        style_loss = tf.reduce_mean(tf.nn.relu(1.0 - style))
        gray_loss = tf.reduce_mean(tf.nn.relu(1.0 + gray))
        fake_loss = tf.reduce_mean(tf.nn.relu(1.0 + fake))
        smooth_loss = tf.reduce_mean(tf.nn.relu(1.0 + smooth))

    loss = style_w * style_loss + \
            fake_w * fake_loss + \
            gray_w * gray_loss  + \
            smooth_w * smooth_loss

    return loss