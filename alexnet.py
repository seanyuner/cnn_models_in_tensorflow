import tensorflow as tf


class AlexNet(object):
    def __init__(self, x, num_classes, keep_prob, regularizer=None,
                 write_sum=False):
        self.x = x
        self.NUM_CLASSES = num_classes
        self.KEEP_PROB = keep_prob
        self.REGULARIZER = regularizer
        self.WRITE_SUM = write_sum
        self.create()


    def variable_summaries(self, name, var):
        tf.summary.histogram(name, var)
            
        mean = tf.reduce_mean(var)
        tf.summary.scalar('%s/mean' % name, mean)
            
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('%s/stddev' % name, stddev)


    def write_summaries(self, weights, biases):
        self.variable_summaries('weights', weights)
        self.variable_summaries('biases', biases)


    def conv(self, base, filter_height, filter_width, num_channels,
             num_filters, stride_y, stride_x, scope_name, padding='SAME'):
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
        if self.WRITE_SUM: self.write_summaries(weights, biases)
        
        conved = tf.nn.conv2d(base, weights, [1, stride_y, stride_x, 1],
                              padding=padding)
        with_bias = tf.nn.bias_add(conved, biases)
        relued = tf.nn.relu(with_bias, name=scope_name)
        
        return relued


    def lrn(self, base, scope_name, depth_radius=2, bias=2.0, alpha=1e-4,
            beta=0.75):
        lrned = tf.nn.local_response_normalization(base, depth_radius,
                                                   bias, alpha,
                                                   beta, name=scope_name)
        return lrned


    def pool(self, base, filter_height, filter_width, stride_y, stride_x,
             scope_name, padding='VALID'):
        pooled = tf.nn.max_pool(base,
                                ksize=[1, filter_height, filter_width, 1],
                                strides=[1, stride_y, stride_x, 1],
                                padding=padding,
                                name=scope_name)
        return pooled


    def fc(self, base, in_nodes, out_nodes, scope_name, relu=True):
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
        if self.WRITE_SUM: self.write_summaries(weights, biases)
        
        fced = tf.nn.xw_plus_b(base, weights, biases, name=scope_name)
        
        if not relu:
            return fced
        
        relued = tf.nn.relu(fced)
        dropout = tf.nn.dropout(relued,
                                keep_prob=self.KEEP_PROB,
                                name=scope_name)
        return dropout
        
        
    def create(self):
        #### layer1 ####
        # conv1
        with tf.name_scope('conv1') as scope:
            self.conv1 = self.conv(self.x, 11, 11, 3, 96, 4, 4, scope)
        # lrn1
        with tf.name_scope('lrn1') as scope:
            self.lrn1 = self.lrn(self.conv1, scope, 2, 2.0, 1e-4, 0.75)
        # pool1
        with tf.name_scope('pool1') as scope:
            self.pool1 = self.pool(self.lrn1, 3, 3, 2, 2, scope)
        
        #### layer2 ####
        # conv2
        with tf.name_scope('conv2') as scope:
            self.conv2 = self.conv(self.pool1, 5, 5, 96, 256, 1, 1, scope)
        # lrn2
        with tf.name_scope('lrn2') as scope:
            self.lrn2 = self.lrn(self.conv2, scope, 2, 2.0, 1e-4, 0.75)
        # pool2
        with tf.name_scope('pool2') as scope:
            self.pool2 = self.pool(self.lrn2, 3, 3, 2, 2, scope)

        #### layer3 ####
        # conv3
        with tf.name_scope('conv3') as scope:
            self.conv3 = self.conv(self.pool2, 3, 3, 256, 384, 1, 1, scope)
            
        #### layer4 ####
        # conv4
        with tf.name_scope('conv4') as scope:
            self.conv4 = self.conv(self.conv3, 3, 3, 384, 384, 1, 1, scope)
            
        #### layer5 ####
        # conv5
        with tf.name_scope('conv5') as scope:
            self.conv5 = self.conv(self.conv4, 3, 3, 384, 256, 1, 1, scope)
        # pool5
        with tf.name_scope('pool5') as scope:
            self.pool5 = self.pool(self.conv5, 3, 3, 2, 2, scope)
            
        #### layer6 ####
        # fc6
        with tf.name_scope('fc6') as scope:
            flattened = tf.reshape(self.pool5, [-1, 6*6*256])
            self.fc6 = self.fc(flattened, 6*6*256, 4096, scope, True)
            
        #### layer7 ####
        # fc7
        with tf.name_scope('fc7') as scope:
            self.fc7 = self.fc(self.fc6, 4096, 4096, scope, True)
            
        #### layer8 ####
        # fc8
        with tf.name_scope('fc8') as scope:
            self.fc8 = self.fc(self.fc7, 4096, self.NUM_CLASSES, scope, False)
