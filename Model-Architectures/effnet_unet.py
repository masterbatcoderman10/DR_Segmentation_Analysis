import tensorflow as tf
from tensorflow.keras.applications.efficientnet import EfficientNetB4, preprocess_input
from .decoders import decoder_block, decoder_full

def effnet_stem(input, layers):

    x = input

    for layer in layers:
        x = layer(x)
    
    return x
    
def effblock_enc(inp, layers, model):

    enc = tf.keras.Model(inputs=model.get_layer(layers[0].name).input, outputs=model.get_layer(layers[-1].name).output)
    x = enc(inp)
    return x

def effnet_encoder(inp, layer_dict, model):

    keys = list(layer_dict.keys())

    x = effnet_stem(inp, layer_dict[keys[0]])

    activations = [x]
    for block in keys[1:]:

        x = effblock_enc(x, layer_dict[block], model) 
        if block == "block_1" or block ==  "block_4" or block == "block_6":
            continue
        activations.append(x)
    
    return x, [None] + activations[:-1]

def effnet_unet(num_classes, input_size, input_dim):

    b4 = EfficientNetB4(weights="imagenet", include_top=False, input_shape=(input_size,input_size,input_dim))

    effnet_blocks_dict = {}
    stem_start = [b4.layers[0], b4.layers[1], b4.layers[2]]
    stem_start.extend([layer for layer in b4.layers if layer.name[:4] == "stem"])
    effnet_blocks_dict["stem"] = stem_start
    effnet_blocks_dict = {**effnet_blocks, **{ 
        f"block_{n}" : [layer for layer in b4.layers if layer.name[:6] == f"block{n}"] for n in range(1, 8)
    }}

    #Freeszing the weights of the model
    b4.trainable = False

    #Encoder of the network
    inp = b4.input
    x = preprocess_input(inp)
    #a not being reversed here because it's reversed in the decoder function
    x, a = effnet_encoder(x, effnet_blocks_dict, b4)

    #Decoder of the network
    filters = [160, 56, 32, 48, 64]
    
    output = decoder_full(a, x, filters, num_classes)

    model  = tf.keras.Model(inp, output)

    return model