import tensorflow as tf


# 定义一个各种CNN模型的父类，其中定义一些通用属性和方法
class CNNs(object):
    def __init__(self, keep_prob, regularizer=None, write_sum=False, 
                 training=True):
        self.KEEP_PROB = keep_prob
        self.REGULARIZER = regularizer
        self.WRITE_LOG = write_sum
        self.TRAINING = training


    # 逐个变量写具体日志
    def variable_summaries(self, name, var):
        tf.summary.histogram(name, var)
            
        mean = tf.reduce_mean(var)
        tf.summary.scalar('%s/mean' % name, mean)
            
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('%s/stddev' % name, stddev)


    # 写日志
    def write_summaries(self, weights, biases):
        self.variable_summaries('weights', weights)
        self.variable_summaries('biases', biases)


    # 自定义卷积层
    def conv(self, base, filter_height, filter_width, num_filters,
             stride_y, stride_x, scope, padding='SAME', batch_norm=False,
             relu=True):
        with tf.name_scope(scope):
            num_channels = base.get_shape()[-1].value
                
            weights = tf.Variable(tf.random_normal([filter_height,
                                                    filter_width,
                                                    num_channels,
                                                    num_filters],
                                                   stddev=1e-2,
                                                   dtype=tf.float32),
                                  name='weights')
            biases = tf.Variable(tf.constant(0.0,
                                             shape=[num_filters],
                                             dtype=tf.float32),
                                 trainable=True,
                                 name='biases')
        
            if self.REGULARIZER != None:
                tf.add_to_collection('losses', self.REGULARIZER(weights))
            if self.WRITE_LOG:
                self.write_summaries(weights, biases)
        
            conved = tf.nn.conv2d(base, weights, [1, stride_y, stride_x, 1],
                                  padding=padding)
            if batch_norm:
                normed = tf.layers.batch_normalization(conved,
                                                       training=self.TRAINING)
                if not relu:
                    return normed
                else:
                    relued = tf.nn.relu(normed, name='relu')
            else:
                with_bias = tf.nn.bias_add(conved, biases)
                if not relu:
                    return with_bias
                else:
                    relued = tf.nn.relu(with_bias, name='relu')
        
            return relued


    # 自定义局部响应归一化
    def lrn(self, base, scope, depth_radius=2, bias=2.0,
            alpha=1e-4, beta=0.75):
        with tf.name_scope(scope):
            lrned = tf.nn.local_response_normalization(base, depth_radius,
                                                       bias, alpha,
                                                       beta, name='lrn')
            return lrned


    
    # 自定义池化层
    def pool(self, base, filter_height, filter_width, stride_y, stride_x,
             scope, padding='VALID', max_pool=True):
        with tf.name_scope(scope):
            if max_pool:
                pooled = tf.nn.max_pool(base,
                                        ksize=[1, filter_height, filter_width, 1],
                                        strides=[1, stride_y, stride_x, 1],
                                        padding=padding,
                                        name='max_pool')
            else:
                pooled = tf.nn.avg_pool(base, 
                                        ksize=[1, filter_height, filter_width, 1],
                                        strides=[1, stride_y, stride_x, 1],
                                        padding=padding,
                                        name='avg_pool')
            return pooled


    # 自定义全连接层
    def fc(self, base, out_nodes, scope, relu=True, reshape=False,
           batch_norm=False):
        with tf.name_scope(scope):
            in_nodes = base.get_shape()[-1].value
            if len(base.get_shape()) != 2:
                for i in range(-3, -1):
                    in_nodes *= base.get_shape()[i].value
                
            if reshape: base = tf.reshape(base, [-1, in_nodes])
            
            weights = tf.Variable(tf.random_normal([in_nodes, out_nodes],
                                                   stddev=1e-2,
                                                   dtype=tf.float32),
                                  name='weights')
         
            biases = tf.Variable(tf.constant(0.0,
                                             shape=[out_nodes],
                                             dtype=tf.float32),
                                 trainable=True,
                                 name='biases')
    
            if self.REGULARIZER != None:
                tf.add_to_collection('losses', self.REGULARIZER(weights))
            if self.WRITE_LOG:
                self.write_summaries(weights, biases)
            
            if batch_norm:
                matmuled = tf.matmul(base, weights, name='matmul')
                normed = tf.layers.batch_normalization(matmuled,
                                                       training=self.TRAINING)
                relued = tf.nn.relu(normed, name='relu')
            else:
                fced = tf.nn.xw_plus_b(base, weights, biases, name='fc')
                if not relu:
                    return fced
                
                relued = tf.nn.relu(fced, name='relu')
                    
            return relued


    # 自定义dropout层
    def dropout(self, base, scope):
        with tf.name_scope(scope):
            dropouted = tf.nn.dropout(base,
                                      keep_prob=self.KEEP_PROB,
                                      name='dropout')
            return dropouted


    # 自定义concat层
    def concat(self, base, axis, scope):
        with tf.name_scope(scope):
            concated = tf.concat(base, axis=3, name='concat')
            return concated
        
    
    # 自定义separable conv层
    def sepa_conv(self, base, num_outputs, kernel_size,
                  depth_multiplier, strides, scope, padding='SAME'):
        with tf.name_scope(scope):
            sepa_conved = tf.contrib.layers.separable_conv2d(base,
                                                             num_outputs,
                                                             [kernel_size, kernel_size],
                                                             depth_multiplier=depth_multiplier,
                                                             stride=strides)
            return sepa_conved
