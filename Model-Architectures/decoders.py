import tensorflow as tf

def decoder_block(a, x, f):

    x = tf.keras.layers.Conv2DTranspose(filters=f, kernel_size=2, strides=2, padding="same", activation="relu")(x)
    x = tf.concat([a, x], axis=-1)
    x = tf.keras.layers.Conv2D(f, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(f, 3, padding="same", activation="relu")(x) 

    return x

def decoder_full(activations, x, filters, num_classes):

    for a,f in zip(activations[::-1],filters):
        x = decoder_block(a, x, f)
    
    output = tf.keras.layers.Conv2D(num_classes, 1, padding="same", activation="softmax")(x)

    return output