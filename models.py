import tensorflow as tf
from utils import normal_initializer, zero_initializer
from layers import ConvLayer, ConvPoolLayer, DeconvLayer
import numpy as np

flags = tf.app.flags
FLAGS = flags.FLAGS


class AutoEncoder(object):
    def __init__(self):
        # placeholder for storing rotated input images
        self.input_rotated_images = tf.placeholder(dtype=tf.float32,
                                                   shape=(None, FLAGS.height, FLAGS.width, FLAGS.num_channel))
        # placeholder for storing original images without rotation
        self.input_original_images = tf.placeholder(dtype=tf.float32,
                                                    shape=(None, FLAGS.height, FLAGS.width, FLAGS.num_channel))

        # self.output_images: images predicted by model
        # self.code_layer: latent code produced in the middle of network
        # self.reconstruct: images reconstructed by model
        self.code_layer, self.reconstruct, self.output_images = self.build()
        self.loss = self._loss()
        self.opt = self.optimization()

    def optimization(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        return optimizer.minimize(self.loss)

    def encoder(self, inputs):


        # Build Convolutional Part of Encoder
        # Put sequential layers:
        #       ConvLayer1 ==> ConvPoolLayer1 ==> ConvLayer2 ==> ConvPoolLayer2 ==> ConvLayer3 ==> ConvPoolLayer3
        # Settings of layers:
        # For all ConvLayers: filter size = 3, filter stride = 1, padding type = SAME
        # For all ConvPoolLayers:
        #   Conv    : filter size = 3, filter stride = 1, padding type = SAME
        #   Pooling :   pool size = 3,   pool stride = 2, padding type = SAME
        # Number of Filters:
        #       num_channel defined in FLAGS (input) ==> 8 ==> 8 ==> 16 ==> 16 ==> 32 ==> 32

        # convolutional layer
        conv1_class = ConvLayer(input_filters=FLAGS.num_channel, output_filters=8, act=tf.nn.relu, 
                                kernel_size=3, kernel_stride=1, kernel_padding='SAME')
        conv1 = conv1_class(inputs=inputs)
        print(conv1.shape)
        # convolutional and pooling layer
        conv_pool1_class = ConvPoolLayer(input_filters=8, output_filters=8, act=tf.nn.relu, 
                                         kernel_size=3, kernel_stride=1, kernel_padding='SAME',
                                         pool_size=3, pool_stride=2, pool_padding='SAME')
        conv_pool1 = conv_pool1_class(inputs=conv1)
        print(conv_pool1.shape)
        # convolutional layer
        conv2_class = ConvLayer(input_filters=8, output_filters=16, act=tf.nn.relu, 
                                kernel_size=3, kernel_stride=1, kernel_padding='SAME')
        conv2 = conv2_class(inputs=conv_pool1)
        print(conv2.shape)
        # convolutional and pooling layer
        conv_pool2_class = ConvPoolLayer(input_filters=16, output_filters=16, act=tf.nn.relu, 
                                         kernel_size=3, kernel_stride=1, kernel_padding='SAME',
                                         pool_size=3, pool_stride=2, pool_padding='SAME')
        conv_pool2 = conv_pool2_class(inputs=conv2)
        print(conv_pool2.shape)

        conv3_class = ConvLayer(input_filters=16, output_filters=32, act=tf.nn.relu, 
                                kernel_size=3, kernel_stride=1, kernel_padding='SAME')
        conv3 = conv3_class(inputs=conv_pool2)
        print(conv3.shape)

        conv_pool3_class = ConvPoolLayer(input_filters=32, output_filters=32, act=tf.nn.relu, 
                                         kernel_size=3, kernel_stride=1, kernel_padding='SAME',
                                         pool_size=3, pool_stride=2, pool_padding='SAME')
        conv_pool3 = conv_pool3_class(inputs=conv3)
        print(conv_pool3.shape)

        # Make Output Flatten and Apply Transformation
        # Num of features for dense is defined by code_size in FLAG

        # make output of pooling flatten
        WholeShape = tf.shape(conv_pool3)
        NumSamples = WholeShape[0]
        last_conv_dims = tf.constant(value=(4,4,32), dtype=tf.int32, shape=(3,))#WholeShape[1:]
        FlattedShape = tf.reduce_prod(last_conv_dims)
        
        flatten = tf.reshape(conv_pool3, shape=[NumSamples, FlattedShape])
        print(flatten.shape)
        
        
        # apply fully connected layer
        W_Trans = normal_initializer(shape=[FlattedShape, FLAGS.code_size])
        B_Trans = zero_initializer(shape=[FLAGS.code_size])
        dense = tf.nn.xw_plus_b(flatten, W_Trans, B_Trans)
        print(dense.shape)

        return dense, last_conv_dims

    def decoder(self, inputs, last_conv_dims):


        # Apply Transformation and Reshape to Original
        # Num of output features in this transformation can be calculated using last_conv_dims
        # Please note that number of input features is code_size stored in FLAGS

        # apply fully connected layer
        FlattedShape = tf.reduce_prod(last_conv_dims)
        
        W_InvTrans = normal_initializer(shape=[FLAGS.code_size, FlattedShape])
        B_InvTrans = zero_initializer(shape=[FlattedShape])
        
        decode_layer = tf.nn.relu(tf.nn.xw_plus_b(inputs, W_InvTrans, B_InvTrans))

        print(decode_layer.shape)
        
        
        
        # reshape to send as input to transposed convolutional layer
        
        deconv_input = tf.reshape(decode_layer, shape=[-1, last_conv_dims[0], last_conv_dims[1], last_conv_dims[2]])
        
        print(deconv_input.shape)

        # Apply 3 Transposed Convolution Sequentially
        # Put sequential layers:
        #       DeconvLayer ==> Deconv Layer ==> Deconv Layer
        # For all layers use:
        #       filter size = 3, filter stride = 2, padding type = SAME
        # Apply tf.nn.relu as activation function for first two layers
        # Multiply all the numbers in last_conv_dims to find num of output features
        # Number of filters:
        #       num_channel defined in FLAGS (input of first deconv) ==> 16 ==> 8 ==> 1

        # transpose convolutional layer
        deconv1_class = DeconvLayer(input_filters=last_conv_dims[2], output_filters=16, act=tf.nn.relu,
                                    kernel_size=3, kernel_stride=2, kernel_padding='SAME')
        deconv1 = deconv1_class(inputs=deconv_input)
        print(deconv1.shape)
        # transpose convolutional layer
        deconv2_class = DeconvLayer(input_filters=16, output_filters=8, act=tf.nn.relu,
                                    kernel_size=3, kernel_stride=2, kernel_padding='SAME')
        deconv2 = deconv2_class(inputs=deconv1)
        print(deconv2.shape)
        # transpose convolutional layer
        deconv3_class = DeconvLayer(input_filters=8, output_filters=1, act=tf.identity,
                                    kernel_size=3, kernel_stride=2, kernel_padding='SAME')
        deconv3 = deconv3_class(inputs=deconv2)
        print(deconv3.shape)

        return deconv3

    def _loss(self):
        # Loss function
        # Apply tf.nn.sigmoid_cross_entropy_with_logits to produce loss
        #       logits: flatten output
        #       labels: flatten input

        NumSamples = tf.shape(self.input_original_images)[0]
        FlattedIn = tf.reshape(self.input_original_images, shape=[NumSamples, -1])
        FlattedRe = tf.reshape(self.reconstruct, shape=[NumSamples, -1])
        mean_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=FlattedIn, logits=FlattedRe))

        return mean_loss

    def build(self):
        # evaluate encoding of images by self.encoder
        code_layer, last_conv_dims = self.encoder(self.input_rotated_images)

        # evaluate reconstructed images by self.decoder
        reconstruct = self.decoder(code_layer, last_conv_dims)

        # apply tf.nn.sigmoid to change pixel range to [0, 1]
        output_images = tf.nn.sigmoid(reconstruct)

        return code_layer, reconstruct, output_images
