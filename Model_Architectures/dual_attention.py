import tensorflow as tf

class SpatialAttention(tf.keras.layers.Layer):

    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.alpha = tf.Variable(initial_value=0.0, trainable=True)
    
    def build(self, input_shape):
        
        self.C = input_shape[-1]
        self.H = input_shape[1]
        self.W = input_shape[2]

        #Defining the convolutions
        self.conv1 = tf.keras.layers.Conv2D(self.C, 1)
        self.conv2 = tf.keras.layers.Conv2D(self.C, 1)
        self.conv3 = tf.keras.layers.Conv2D(self.C, 1)

    def call(self, inputs):

        n_shape = self.H * self.W

        a = inputs
        b = self.conv1(inputs)
        c = self.conv2(inputs)
        d = self.conv3(inputs)

        b = tf.transpose(tf.keras.layers.Reshape((n_shape, self.C))(b), perm=[0,2,1])
        c = tf.keras.layers.Reshape((n_shape, self.C))(c)
        d = tf.keras.layers.Reshape((n_shape, self.C))(d)

        c = tf.linalg.matmul(c, b)
        S = tf.keras.layers.Softmax()(c)
        S = tf.transpose(S, perm=[0,2,1])

        d = self.alpha * tf.linalg.matmul(S, d)
        d = tf.keras.layers.Reshape((self.H, self.W, self.C))(d)
        E = tf.keras.layers.Add()([a, d])        

        return E

class ChannelAttention(tf.keras.layers.Layer):

    def __init__(self):
        super(ChannelAttention, self).__init__()
        self.beta = tf.Variable(initial_value=0.0, name="beta", trainable=True)
    
    def build(self, input_shape):
        self.C = input_shape[-1]
        self.H = input_shape[1]
        self.W = input_shape[2]
    
    def call(self, inputs):

        a1=a2=a3=a4= inputs
        n_shape = self.H * self.W
        a2 = tf.keras.layers.Reshape((n_shape, self.C))(a2)
        a3 = tf.keras.layers.Reshape((n_shape, self.C))(a3)
        a4 = tf.transpose(tf.keras.layers.Reshape((n_shape, self.C))(a4), perm=[0,2,1])


        #Creating X, the softmax on the matrix product of A_T_A
        a_T_a = tf.linalg.matmul(a4, a3)
        x = tf.keras.layers.Softmax()(a_T_a)
        x = tf.transpose(x, perm=[0,2,1])

        a2_pass = self.beta * tf.linalg.matmul(a2, x)
        a2_pass = tf.keras.layers.Reshape((self.H,self.W,self.C))(a2_pass)

        E = tf.keras.layers.Add()([a1, a2_pass])

        return E
            

class DualAttention(tf.keras.layers.Layer):

    def __init__(self):
        super(DualAttention, self).__init__()
    
    def build(self, input_shape):
        self.C = input_shape[-1]
        self.conv1 = tf.keras.layers.Conv2D(self.C, 1)
        self.conv2 = tf.keras.layers.Conv2D(self.C, 1)
        self.sam = SpatialAttention()
        self.cam = ChannelAttention()
    
    def call(self, inputs):

        e1 = self.sam(inputs)
        e2 = self.cam(inputs)

        e1 = self.conv1(e1)
        e2 = self.conv2(e2)

        F = tf.keras.layers.Add()([e1, e2])
        return F

