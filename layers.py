import tensorflow as tf
from utils import zero_initializer, normal_initializer


class ConvLayer(object):
    def __init__(self, input_filters, output_filters, act,
                 kernel_size, kernel_stride, kernel_padding):

        super(ConvLayer, self).__init__()

        # number of input channels
        self.input_filters = input_filters

        # number of output channels
        self.output_filters = output_filters

        # convolutional filters kernel size
        self.kernel_size = kernel_size

        # stride of convolutional filters
        self.kernel_stride = kernel_stride

        # padding type of filters
        self.kernel_padding = kernel_padding

        # activation function type
        self.act = act

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):

        # Define Filters and Bias
        # Define filter tensor with proper size using normal initializer
        # Define bias tensor as well using zero initializer

        self.conv_filter = normal_initializer(shape=[self.kernel_size, self.kernel_size, self.input_filters, self.output_filters])
        self.conv_bias = zero_initializer(shape=[self.output_filters])



        # Apply Convolution, Bias and Activation Function
        # Use tf.nn.conv2d and give it following inputs
        #   1. Input tensor
        #   2. Filter you have defined in above empty part
        #   3. Stride tensor showing stride size for each dimension
        #   4. Padding type based on self.kernel_padding

        self.conv_output = tf.nn.conv2d(inputs, self.conv_filter, strides=[1, self.kernel_stride, self.kernel_stride, 1], padding=self.kernel_padding)
        # Add bias and apply activation
        self.total_output = self.act(tf.add(self.conv_output, self.conv_bias))

        return self._call(self.total_output)


class ConvPoolLayer(ConvLayer):
    def __init__(self, input_filters, output_filters, act,
                 kernel_size, kernel_stride, kernel_padding,
                 pool_size, pool_stride, pool_padding):

        # Calling ConvLayer constructor will store convolutional section config
        super(ConvPoolLayer, self).__init__(input_filters, output_filters, act,
                                            kernel_size, kernel_stride, kernel_padding)

        # size of kernel in pooling
        self.pool_size = pool_size

        # size of stride in pooling
        self.pool_stride = pool_stride

        # type of padding in pooling
        self.pool_padding = pool_padding

    def _call(self, inputs):

        # Apply Pooling
        # Please note that when __call__ method is called for an object of this
        # class, convolution operation will be applied on original input.
        # We override _call function so that the result convolution will later
        # move through max pooling function which should be defined below.
        # To do so, use tf.nn.max_pool and give it following inputs:
        #   1. Input tensor
        #   2. Kernel size for max pooling
        #   3. Stride tensor showing stride size for each dimension
        #   4. Padding type based on self.kernel_padding

        self.pooling_output = tf.nn.max_pool(inputs, ksize=[1, self.pool_size, self.pool_size, 1], strides=[1, self.pool_stride, self.pool_stride, 1], padding=self.pool_padding)

        return self.pooling_output


class DeconvLayer(object):
    def __init__(self, input_filters, output_filters, act,
                 kernel_size, kernel_stride, kernel_padding):

        super(DeconvLayer, self).__init__()

        # number of input channels
        self.input_filters = input_filters

        # number of output channels
        self.output_filters = output_filters

        # transposed convolutional filters kernel size
        self.kernel_size = kernel_size

        # stride of transposed convolutional filters
        self.kernel_stride = kernel_stride

        # padding type of filters
        self.kernel_padding = kernel_padding

        # activation function type
        self.act = act

    def __call__(self, inputs):

        # Define Filters and Bias
        # Note that tensor shape of this filter is different from that of the filter in ConvLayer

        self.deconv_filter = normal_initializer(shape=[self.kernel_size, self.kernel_size, self.output_filters, self.input_filters])
        self.deconv_bias = zero_initializer(shape=[self.output_filters])

        # input height and width
        input_height = inputs.get_shape().as_list()[1]
        input_width = inputs.get_shape().as_list()[2]


        # Calculate Output Shape
        # The formula to calculate output shapes depends on type of padding

        if self.kernel_padding == 'SAME':
            output_height = self.kernel_stride*input_height
            output_width = self.kernel_stride*input_width
        elif self.kernel_padding == 'VALID':
            output_height = self.kernel_stride*(input_height-1)+self.kernel_size
            output_width = self.kernel_stride*(input_width-1)+self.kernel_size
        else:
            raise Exception('No such padding')


        # Apply Transposed Convolution, Bias and Activation Function
        # Use tf.nn.conv2d_transpose and give it following inputs
        #   1. Input tensor
        #   2. Filter you have defined above
        #   3. Output shape you have calculated above
        #   4. Stride tensor showing stride size for each dimension
        #   5. Padding type based on self.kernel_padding

        batch_size = tf.shape(inputs)[0]
        Depth = inputs.get_shape().as_list()[3]
        deconv_shape = tf.stack([batch_size, output_height, output_width, self.output_filters])
        
        self.deconv_output = tf.nn.conv2d_transpose(inputs, self.deconv_filter, output_shape=deconv_shape, strides=[1, self.kernel_stride, self.kernel_stride, 1], padding=self.kernel_padding)
        # Add bias and apply activation
        self.total_output = self.act(tf.add(self.deconv_output, self.deconv_bias))

        return self.total_output
