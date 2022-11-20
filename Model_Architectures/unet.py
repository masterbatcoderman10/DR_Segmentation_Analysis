import tensorflow as tf
from .decoders import decoder_block, decoder_full

def encoder_block(inp,f):

    x = inp

    x = tf.keras.layers.Conv2D(f, 3, 1, padding="same")(x)
    x = tf.keras.layers.Conv2D(f, 3, 1, padding="same")(x)
    a = x
    x = tf.keras.layers.MaxPool2D(2, 2)(x)

    return x,a

def last_encoder(inp, f):

    x = inp

    x = tf.keras.layers.Conv2D(f, 3, 1, padding="same")(x)
    x = tf.keras.layers.Conv2D(f, 3, 1, padding="same")(x)

    return x

def encoder_unet(inp, filters):

    activations = []

    x = inp
    for f in filters[:-1]:
        x, a = encoder_block(x, f)
        activations.append(a)
    
    x = last_encoder(x, filters[-1])
    return x, activations

def unet(num_classes, input_size, input_dim):

    inp = tf.keras.layers.Input((input_size, input_size, input_dim))

    filters = [64, 128, 256, 512, 1024]

    x, a = encoder_unet(inp, filters=filters)

    o = decoder_full(a, x, filters, num_classes)

    model = tf.keras.Model(inp, o)

    return model