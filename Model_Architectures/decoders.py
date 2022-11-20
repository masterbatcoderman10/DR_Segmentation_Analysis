import tensorflow as tf
from ..Modules.dual_attention import DualAttention, SpatialAttention, ChannelAttention

def decoder_block(a, x, f, attention=False):

    if attention:
        a = DualAttention()(a)

    x = tf.keras.layers.Conv2DTranspose(filters=f, kernel_size=2, strides=2, padding="same", activation="relu")(x)
    if a is not  None:
        x = tf.concat([a, x], axis=-1)
    x = tf.keras.layers.Conv2D(f, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.Conv2D(f, 3, padding="same", activation="relu")(x) 

    return x

def decoder_full(activations, x, filters, num_classes, attention_indices):

    #Looping over the activations and filters from bottom to top
    #Activation are reversed for this effect


        
    
    ai = None
    #Flag to indicate whether the point of attention is found
    found = True
    #Flag to pass to the decoder block, whether dual attention should be applied
    att = False
    for i,(a,f) in enumerate(zip(activations[::-1],filters)):

        #Flag to indicate whether there is no need for attention
        there = len(attention_indices)
        if found and there:
            ai = attention_indices.pop()
        
        #Check if the current activation needs attention

        att=found = (i+1 == ai)
        print(att)
        x = decoder_block(a, x, f, att)
    
    output = tf.keras.layers.Conv2D(num_classes, 1, padding="same", activation="softmax")(x)

    return output