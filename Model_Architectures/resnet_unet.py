import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.resnet import ResNet50, preprocess_input
from .decoders import decoder_block, decoder_full
from ..Modules.dual_attention import DualAttention, SpatialAttention, ChannelAttention

def resblock_stem(input, layers):

    x = input
    for layer in layers:

        x = layer(x)
    
    a = x
    x = tf.keras.layers.MaxPool2D(2, 2)(x)

    return x, a


def resblock_enc(inp, layers, res):

    enc = tf.keras.Model(inputs=res.get_layer(layers[0].name).input, outputs=res.get_layer(layers[-1].name).output)
    x = enc(inp)
    return x

def resnet_encoder(inp, layer_dict, res):

    keys = list(layer_dict.keys())

    x, a = resblock_stem(inp, layer_dict[keys[0]])

    activations = [a]
    for block in keys[1:]:
        x = resblock_enc(x, layer_dict[block], res) 
        activations.append(x)
    
    return x, [None] + activations[:-1]



def resnet_unet(num_classes, input_size, input_dim, att_indices=[], last_attention=False):

    #Downloading the ResNet
    resnet = ResNet50(weights="imagenet", include_top=False, input_shape=(input_size,input_size,input_dim))

    layer_dict = resnet_blocks = {
    f"block_{n}" : [layer for layer in resnet.layers if layer.name[:5] == f"conv{n}"] for n in range(1, 5)
    }

    #Freezing the layers of the ResNet
    resnet.trainable = False

    #Building the model
    inp = resnet.input
    x = preprocess_input(x=inp)
    x, a = resnet_encoder(inp, layer_dict, resnet)


    #Decoder
    filters = [512, 256, 128, 64]

    l = len(att_indices)
    assert l >= 0, "Attention indices should be 0 or greater"
    assert l <= len(filters) + 1, "Number of layers for attetention can not exceed 5"

    #assert len(att_indices[att_indices > 5]) == 0, "Attention indices must be from 1 to 5"

    if last_attention:
        x = DualAttention()(x)

    output = decoder_full(a, x, filters, num_classes, att_indices)

    model = tf.keras.Model(inp, output)

    return model