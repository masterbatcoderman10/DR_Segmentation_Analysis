import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from .decoders import decoder_block, decoder_full
from ..Modules.dual_attention import DualAttention, SpatialAttention, ChannelAttention

def vgg_encoder_block(x, layers):
    """
    This function passes an input through a set of conv layers from VGG19, returning the downsampled and convolved activation
    """
    for layer in layers:
        x = layer(x)
    
    addition = x
    x = tf.keras.layers.MaxPooling2D((2,2), strides = 2)(x)
    return (x, addition)

def last_vgg_block(x, layers):

    for layer in layers:
        x = layer(x)
    
    return x

def vgg_encoder_full(input, layer_dict):

    """
    This function creates the full encoder given a dictionary of layers from the VGG network, it returns the final activation 
    and a list of intermediate activations
    """

    activations = []
    x = input
    for layer_name in list(layer_dict.keys())[:-1]:
        x, a = vgg_encoder_block(x, layer_dict[layer_name])
        activations.append(a)
    
    x = last_vgg_block(x, layer_dict[list(layer_dict.keys())[-1]])
    
    return x, activations



def vgg_unet(num_classes, input_size, input_dim, att_indices=[], last_attention=False):

    #Downloading the VGG network
    vgg19 = VGG19(weights="imagenet", include_top=False, input_shape=(input_size, input_size,input_dim))
    vgg19.trainable = False
    #Getting all the blocks from the VGG network
    vgg_blocks = {
        f"block{n}" : [layer for layer in vgg19.layers if f"block{n}_conv" in layer.name] for n in range(1, 6)
    }
    
    #Filters for the Decoder
    filters = [512, 256, 128, 64]

    l = len(att_indices)
    assert l >= 0, "Attention indices should be 0 or greater"
    assert l <= len(filters) + 1, "Number of layers for attetention can not exceed 5"

    #assert len(att_indices[att_indices > 5]) == 0, "Attention indices must be from 1 to 5"
    

    vgg_input = vgg19.input

    #Defining the encoder

    #First Preprocess the input
    x = preprocess_input(x=vgg_input)
    
    x, a = vgg_encoder_full(x, vgg_blocks)

    if last_attention:
        x = DualAttention()(x)

    output = decoder_full(a, x, filters, num_classes, att_indices)

    vgg_unet_model = tf.keras.Model(vgg_input, output)

    return vgg_unet_model
    

    

    