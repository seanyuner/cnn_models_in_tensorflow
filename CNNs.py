import tensorflow as tf


# 定义一个各种CNN模型的父类，其中定义一些通用属性和方法
class CNNs(object):
    def __init__(self, keep_prob, regularizer=None, write_sum=False):
        self.KEEP_PROB = keep_prob
        self.REGULARIZER = regularizer
        self.WRITE_LOG = write_sum


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
             stride_y, stride_x, scope, padding='SAME'):
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
            with_bias = tf.nn.bias_add(conved, biases)
            relued = tf.nn.relu(with_bias, name='relu')
        
            return relued


    # 自定义局部响应归一化
    def lrn(self, base, scope, depth_radius=2, bias=2.0, alpha=1e-4,beta=0.75):
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
    def fc(self, base, out_nodes, scope, relu=True, reshape=False):
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

    
# 定义AlexNet模型
class AlexNet(CNNs):
    def __init__(self, x, num_classes, keep_prob, regularizer=None,
                 write_sum=False):
        super().__init__(keep_prob, regularizer, write_sum)
        self.x = x
        self.NUM_CLASSES = num_classes
        self.create()


    def create(self):
        #### layer1 ####
        self.conv1 = self.conv(self.x, 11, 11, 96, 4, 4, 'conv1', 'SAME')
        self.lrn1 = self.lrn(self.conv1, 'lrn1', 2, 2.0, 1e-4, 0.75)
        self.pool1 = self.pool(self.lrn1, 3, 3, 2, 2, 'pool1')
        
        #### layer2 ####
        self.conv2 = self.conv(self.pool1, 5, 5, 256, 1, 1, 'conv2', 'SAME')
        self.lrn2 = self.lrn(self.conv2, 'lrn2', 2, 2.0, 1e-4, 0.75)
        self.pool2 = self.pool(self.lrn2, 3, 3, 2, 2, 'pool2')

        #### layer3 ####
        self.conv3 = self.conv(self.pool2, 3, 3, 384, 1, 1, 'conv3', 'SAME')
            
        #### layer4 ####
        self.conv4 = self.conv(self.conv3, 3, 3, 384, 1, 1, 'conv4', 'SAME')
            
        #### layer5 ####
        self.conv5 = self.conv(self.conv4, 3, 3, 256, 1, 1, 'conv5', 'SAME')
        self.pool3 = self.pool(self.conv5, 3, 3, 2, 2, 'pool3')
            
        #### layer6 ####
        self.fc6 = self.fc(self.pool3, 4096, 'fc6', True, True)
        self.dropout1 = self.dropout(self.fc6, 'dropout1')
            
        #### layer7 ####
        self.fc7 = self.fc(self.dropout1, 4096, 'fc7', True)
        self.dropout2 = self.dropout(self.fc7, 'dropout2')
            
        #### layer8 ####
        self.last = self.fc(self.dropout2, self.NUM_CLASSES, 'fc8', False)


# 定义Vgg16模型
class Vgg16(CNNs):
    def __init__(self, x, num_classes, keep_prob, regularizer=None,
                 write_sum=False):
        super().__init__(keep_prob, regularizer, write_sum)
        self.x = x
        self.NUM_CLASSES = num_classes
        self.create()
        
        
    def create(self):
        #### block1 ####
        self.conv1_1 = self.conv(self.x, 3, 3, 64, 1, 1, 'block1_conv1', 'SAME')
        self.conv1_2 = self.conv(self.conv1_1, 3, 3, 64, 1, 1, 'block1_conv2', 'SAME')
        self.pool1_1 = self.pool(self.conv1_2, 2, 2, 2, 2, 'block1_pool1', 'SAME')
            
        #### block2 ####
        self.conv2_1 = self.conv(self.pool1_1, 3, 3, 128, 1, 1, 'block2_conv1', 'SAME')
        self.conv2_2 = self.conv(self.conv2_1, 3, 3, 128, 1, 1, 'block2_conv2', 'SAME')
        self.pool2_1 = self.pool(self.conv2_2, 2, 2, 2, 2, 'block2_pool1', 'SAME')
            
        #### block3 ####
        self.conv3_1 = self.conv(self.pool2_1, 3, 3, 256, 1, 1, 'block3_conv1', 'SAME')
        self.conv3_2 = self.conv(self.conv3_1, 3, 3, 256, 1, 1, 'block3_conv2', 'SAME')
        self.conv3_3 = self.conv(self.conv3_2, 3, 3, 256, 1, 1, 'block3_conv3', 'SAME')
        self.pool3_1 = self.pool(self.conv3_3, 2, 2, 2, 2, 'block3_pool1', 'SAME')
        
        #### block4 ####
        self.conv4_1 = self.conv(self.pool3_1, 3, 3, 512, 1, 1, 'block4_conv1', 'SAME')
        self.conv4_2 = self.conv(self.conv4_1, 3, 3, 512, 1, 1, 'block4_conv2', 'SAME')
        self.conv4_3 = self.conv(self.conv4_2, 3, 3, 512, 1, 1, 'block4_conv3', 'SAME')
        self.pool4_1 = self.pool(self.conv4_3, 2, 2, 2, 2, 'block4_pool1', 'SAME')
            
        #### block5 ####
        self.conv5_1 = self.conv(self.pool4_1, 3, 3, 512, 1, 1, 'block5_conv1', 'SAME')
        self.conv5_2 = self.conv(self.conv5_1, 3, 3, 512, 1, 1, 'block5_conv2', 'SAME')
        self.conv5_3 = self.conv(self.conv5_2, 3, 3, 512, 1, 1, 'block5_conv3', 'SAME')
        self.pool5_1 = self.pool(self.conv5_3, 2, 2, 2, 2,  'block5_pool1', 'SAME')
            
        #### fc ####
        self.fc1 = self.fc(self.pool5_1, 4096, 'fc_fc1', True, True)
        self.fc2 = self.fc(self.fc1, 4096, 'fc_fc2', True)
        self.last = self.fc(self.fc2, self.NUM_CLASSES, 'fc_fc3', True)
#             self.softmax = tf.nn.softmax(self.fc3)    # 这一步放在损失函数时用tf.nn.softmax...


# 定义Vgg19模型
class Vgg19(CNNs):
    def __init__(self, x, num_classes, keep_prob, regularizer=None,
                 write_sum=False):
        super().__init__(keep_prob, regularizer, write_sum)
        self.x = x
        self.NUM_CLASSES = num_classes
        self.create()
        
        
    def create(self):
        #### block1 ####
        self.conv1_1 = self.conv(self.x, 3, 3, 64, 1, 1, 'block1_conv1', 'SAME')
        self.conv1_2 = self.conv(self.conv1_1, 3, 3, 64, 1, 1, 'block1_conv2', 'SAME')
        self.pool1_1 = self.pool(self.conv1_2, 2, 2, 2, 2, 'block1_pool1', 'SAME')
            
        #### block2 ####
        self.conv2_1 = self.conv(self.pool1_1, 3, 3, 128, 1, 1, 'block2_conv1', 'SAME')
        self.conv2_2 = self.conv(self.conv2_1, 3, 3, 128, 1, 1, 'block2_conv2', 'SAME')
        self.pool2_1 = self.pool(self.conv2_2, 2, 2, 2, 2, 'block2_pool1', 'SAME')
            
        #### block3 ####
        self.conv3_1 = self.conv(self.pool2_1, 3, 3, 256, 1, 1, 'block3_conv1', 'SAME')
        self.conv3_2 = self.conv(self.conv3_1, 3, 3, 256, 1, 1, 'block3_conv2', 'SAME')
        self.conv3_3 = self.conv(self.conv3_2, 3, 3, 256, 1, 1, 'block3_conv3', 'SAME')
        self.conv3_4 = self.conv(self.conv3_3, 3, 3, 256, 1, 1, 'block3_conv4', 'SAME')
        self.pool3_1 = self.pool(self.conv3_4, 2, 2, 2, 2, 'block3_pool1', 'SAME')
        
        #### block4 ####
        self.conv4_1 = self.conv(self.pool3_1, 3, 3, 512, 1, 1, 'block4_conv1', 'SAME')
        self.conv4_2 = self.conv(self.conv4_1, 3, 3, 512, 1, 1, 'block4_conv2', 'SAME')
        self.conv4_3 = self.conv(self.conv4_2, 3, 3, 512, 1, 1, 'block4_conv3', 'SAME')
        self.conv4_4 = self.conv(self.conv4_3, 3, 3, 512, 1, 1, 'block4_conv4', 'SAME')
        self.pool4_1 = self.pool(self.conv4_4, 2, 2, 2, 2, 'block4_pool1', 'SAME')
            
        #### block5 ####
        self.conv5_1 = self.conv(self.pool4_1, 3, 3, 512, 1, 1, 'block5_conv1', 'SAME')
        self.conv5_2 = self.conv(self.conv5_1, 3, 3, 512, 1, 1, 'block5_conv2', 'SAME')
        self.conv5_3 = self.conv(self.conv5_2, 3, 3, 512, 1, 1, 'block5_conv3', 'SAME')
        self.conv5_4 = self.conv(self.conv5_3, 3, 3, 512, 1, 1, 'block5_conv4', 'SAME')
        self.pool5_1 = self.pool(self.conv5_4, 2, 2, 2, 2,  'block5_pool1', 'SAME')
            
        #### fc ####
        self.fc1 = self.fc(self.pool5_1, 4096, 'fc_fc1', True, True)
        self.fc2 = self.fc(self.fc1, 4096, 'fc_fc2', True)
        self.last = self.fc(self.fc2, self.NUM_CLASSES, 'fc_fc3', True)
#             self.softmax = tf.nn.softmax(self.fc3)    # 这一步放在损失函数时用tf.nn.softmax...


# 定义GoogLeNet模型
class GoogLeNet(CNNs):
    def __init__(self, x, num_classes, keep_prob, regularizer=None,
                 write_sum=False, auxil=False):
        super().__init__(keep_prob, regularizer, write_sum)
        self.x = x
        self.NUM_CLASSES = num_classes
        self.AUXIL = auxil
        self.create()
        

    # inception module
    def inception(self, base, N_1x1, N_3x3_reduce, N_3x3, N_5x5_reduce, N_5x5, N_pool_proj, 
                  k_1x1, k_3x3_reduce, k_3x3, k_5x5_reduce, k_5x5, k_pool, k_pool_proj, 
                  s_1x1, s_3x3_reduce, s_3x3, s_5x5_reduce, s_5x5, s_pool, s_pool_proj, scope):
        with tf.name_scope(scope):
            
            inception_1x1 = self.conv(base, k_1x1, k_1x1, N_1x1, s_1x1, s_1x1, '1x1', 'SAME')
            
            inception_3x3_reduce = self.conv(base, k_3x3_reduce, k_3x3_reduce, N_3x3_reduce,
                                             s_3x3_reduce, s_3x3_reduce, '3x3_reduce',  'SAME')
            inception_3x3 = self.conv(inception_3x3_reduce, k_3x3, k_3x3, N_3x3,
                                      s_3x3, s_3x3, '3x3', 'SAME')
            
            inception_5x5_reduce = self.conv(base, k_5x5_reduce, k_5x5_reduce, N_5x5_reduce,
                                             s_5x5_reduce, s_5x5_reduce, '5x5_reduce', 'SAME')
            inception_5x5 = self.conv(inception_5x5_reduce, k_5x5, k_5x5, N_5x5,
                                      s_5x5, s_5x5, '5x5', 'SAME')
            
            inception_pool = self.pool(base, k_pool, k_pool, s_pool, s_pool, 'pool', 'SAME')
            inception_pool_proj = self.conv(inception_pool, k_pool_proj, k_pool_proj, N_pool_proj,
                                            s_pool_proj, s_pool_proj, 'pool_proj', 'SAME')
            
            inception_concat = self.concat([inception_1x1, inception_3x3,
                                            inception_5x5, inception_pool_proj],
                                           3, 'concat')
            return inception_concat


    # auxiliary classifier
    def auxil(self, base, N_conv, N_fc1, k_pool, k_conv, s_pool, s_conv, scope):
        with tf.name_scope(scope):
            auxil_pool = self.pool(base, k_pool, k_pool, s_pool, s_pool, 'pool', 'VALID', max_pool=False)
            auxil_conv = self.conv(auxil_pool, k_conv, k_conv, N_conv, s_conv, s_conv, 'conv', 'SAME')
            auxil_fc1 = self.fc(auxil_conv, N_fc1, 'fc1', True, True)
            auxil_dropout = self.dropout(auxil_fc1, 'dropout')
            auxil_fc2 = self.fc(auxil_dropout, self.NUM_CLASSES, 'fc2', True)
            return auxil_fc2
        
        
    def create(self):
        self.conv1 = self.conv(self.x, 7, 7, 64, 2, 2, 'conv1', 'SAME')
        self.pool1 = self.pool(self.conv1, 3, 3, 2, 2, 'pool1', 'SAME')
        self.lrn1 = self.lrn(self.pool1, 'lrn1', 2, 2.0, 1e-4, 0.75)

        self.conv2_reduce = self.conv(self.lrn1, 1, 1, 64, 1, 1, 'conv2_reduce', 'VALID')
        self.conv2 = self.conv(self.conv2_reduce, 3, 3, 192, 1, 1, 'conv2', 'SAME')
        self.lrn2 = self.lrn(self.conv2, 'lrn2', 2, 2.0, 1e-4, 0.75)
        self.pool2 = self.pool(self.lrn2, 3, 3, 2, 2, 'pool2', 'SAME')
        
        self.inception3a_concat = self.inception(self.pool2, 64, 96, 128, 16, 32, 32, 
                                                 1, 1, 3, 1, 5, 3, 1, 
                                                 1, 1, 1, 1, 1, 1, 1, 'inception3a')
        self.inception3b_concat = self.inception(self.inception3a_concat, 64, 96, 128, 16, 32, 32, 
                                                 1, 1, 3, 1, 5, 3, 1,
                                                 1, 1, 1, 1, 1, 1, 1, 'inception3b')
        
        self.pool3 = self.pool(self.inception3b_concat, 3, 3, 2, 2, 'pool3', 'SAME')
        
        self.inception4a_concat = self.inception(self.pool3, 192, 96, 208, 16, 48, 64, 
                                                 1, 1, 3, 1, 5, 3, 1, 
                                                 1, 1, 1, 1, 1, 1, 1, 'inception4a')
        self.inception4b_concat = self.inception(self.inception4a_concat, 160, 112, 224, 24, 64, 64, 
                                                 1, 1, 3, 1, 5, 3, 1,
                                                 1, 1, 1, 1, 1, 1, 1, 'inception4b')
        self.inception4c_concat = self.inception(self.inception4b_concat, 128, 128, 256, 24, 64, 64, 
                                                 1, 1, 3, 1, 5, 3, 1, 
                                                 1, 1, 1, 1, 1, 1, 1, 'inception4c')
        self.inception4d_concat = self.inception(self.inception4c_concat, 112, 144, 288, 32, 64, 64, 
                                                 1, 1, 3, 1, 5, 3, 1,
                                                 1, 1, 1, 1, 1, 1, 1, 'inception4d')
        self.inception4e_concat = self.inception(self.inception4d_concat, 256, 160, 320, 32, 128, 128, 
                                                 1, 1, 3, 1, 5, 3, 1,
                                                 1, 1, 1, 1, 1, 1, 1, 'inception4e')
        
        self.pool4 = self.pool(self.inception4e_concat, 3, 3, 2, 2, 'pool4', 'SAME')
        
        self.inception5a_concat = self.inception(self.pool4, 256, 160, 320, 32, 128, 128, 
                                                 1, 1, 3, 1, 5, 3, 1, 
                                                 1, 1, 1, 1, 1, 1, 1, 'inception5a')
        self.inception5b_concat = self.inception(self.inception5a_concat, 384, 192, 384, 48, 128, 128, 
                                                 1, 1, 3, 1, 5, 3, 1,
                                                 1, 1, 1, 1, 1, 1, 1, 'inception5b')
        
        self.pool5 = self.pool(self.inception5b_concat, 7, 7, 1, 1, 'pool5', 'VALID', max_pool=False)

        self.dropout1 = self.dropout(self.pool5, 'dropout')

        self.last = self.fc(self.dropout1, self.NUM_CLASSES, 'fc', True, True)
#        self.softmax = tf.nn.softmax(self.fc)    # 这一步放在损失函数时用tf.nn.softmax...（下同）

        if self.AUXIL is not None:
            self.auxil1_last = self.auxil(self.inception4a_concat, 128, 1024, 5, 1, 3, 1, 'auxil1')
            self.auxil2_last = self.auxil(self.inception4d_concat, 128, 1024, 5, 1, 3, 1, 'auxil2')
            self.last = tf.reduce_mean([self.last, 0.3 * self.auxil1_last, 0.3 * self.auxil2_last],
                                       axis=0, keepdims=True)
