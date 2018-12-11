import tensorflow as tf
from models.base import CNNs

# 定义inception_v2模型
class Inception_v2(CNNs):
    def __init__(self, x, num_classes, keep_prob, regularizer=None,
                 write_sum=False, training=True):
        super().__init__(keep_prob, regularizer, write_sum, training)
        self.x = x
        self.NUM_CLASSES = num_classes
        self.create()
        
        
    # inception module
    def inception(self, base, N_1x1, N_3x3_reduce, N_3x3, N_d3x3_reduce, N_d3x3_1, N_d3x3_2, N_pool_proj, 
                  scope, max_pool=False, pass_through=False):
        with tf.name_scope(scope):
            if not pass_through:
                inception_1x1 = self.conv(base, 1, 1, N_1x1, 1, 1, '1x1', batch_norm=True)
                s_3x3, s_d3x3_2, s_pool = 1, 1, 1
            else:
                s_3x3, s_d3x3_2, s_pool = 2, 2, 2
            
            inception_3x3_reduce = self.conv(base, 1, 1, N_3x3_reduce, 1, 1, '3x3_reduce',  batch_norm=True)
            inception_3x3 = self.conv(inception_3x3_reduce, 3, 3, N_3x3, s_3x3, s_3x3, '3x3', batch_norm=True)
            
            inception_d3x3_reduce = self.conv(base, 1, 1, N_d3x3_reduce, 1, 1, 'd3x3_reduce', batch_norm=True)
            inception_d3x3_1 = self.conv(inception_d3x3_reduce, 3, 3, N_d3x3_1, 1, 1, 'd3x3_1', batch_norm=True)
            inception_d3x3_2 = self.conv(inception_d3x3_1, 3, 3, N_d3x3_2, s_d3x3_2, s_d3x3_2, 'd3x3_2', batch_norm=True)
            
            inception_pool = self.pool(base, 3, 3, s_pool, s_pool, 'pool', 'SAME', max_pool)
            if not pass_through:
                inception_pool_proj = self.conv(inception_pool, 1, 1, N_pool_proj, 1, 1, 'pool_proj', batch_norm=True)
                inception_concat = self.concat([inception_1x1, inception_3x3, 
                                                inception_d3x3_2, inception_pool_proj], 
                                               3, 'concat')
            else:
                inception_concat = self.concat([inception_3x3, inception_d3x3_2, inception_pool], 
                                               3, 'concat')
            return inception_concat

        
    def create(self):
        self.conv1 = self.sepa_conv(self.x, 64, 7, 8, 2, 'conv1')
        self.pool1 = self.pool(self.conv1, 3, 3, 2, 2, 'pool1', 'SAME')
        self.lrn1 = self.lrn(self.pool1, 'lrn1', 2, 2.0, 1e-4, 0.75)

        self.conv2_reduce = self.conv(self.lrn1, 1, 1, 64, 1, 1, 'conv2_reduce', 'VALID', batch_norm=True)
        self.conv2 = self.conv(self.conv2_reduce, 3, 3, 192, 1, 1, 'conv2', batch_norm=True)
        self.lrn2 = self.lrn(self.conv2, 'lrn2', 2, 2.0, 1e-4, 0.75)
        self.pool2 = self.pool(self.lrn2, 3, 3, 2, 2, 'pool2', 'SAME')
        
        self.inception3a_concat = self.inception(self.pool2, 64, 64, 64, 64, 96, 96, 32,  'inception3a')
        self.inception3b_concat = self.inception(self.inception3a_concat, 64, 64, 96, 64, 96, 96, 64, 'inception3b')
        self.inception3c_concat = self.inception(self.inception3b_concat, None, 128, 160, 64, 96, 96, None, 'inception3c', True, True)
        
        self.inception4a_concat = self.inception(self.inception3c_concat, 224, 64, 96, 96, 128, 128, 128,  'inception4a')
        self.inception4b_concat = self.inception(self.inception4a_concat, 192, 96, 128, 96, 128, 128, 128, 'inception4b')
        self.inception4c_concat = self.inception(self.inception4b_concat, 160, 128, 160, 128, 160, 160, 128, 'inception4c')
        self.inception4d_concat = self.inception(self.inception4c_concat, 96, 128, 192, 160, 192, 192, 128, 'inception4d')
        self.inception4e_concat = self.inception(self.inception4d_concat, None, 128, 192, 192, 256, 256, None, 'inception4e', True, True)
        
        self.inception5a_concat = self.inception(self.inception4e_concat, 352, 192, 320, 160, 224, 224, 128, 'inception5a')
        self.inception5b_concat = self.inception(self.inception5a_concat, 352, 192, 320, 192, 224, 224, 128, 'inception5b', True)
        
        self.pool3 = self.pool(self.inception5b_concat, 7, 7, 1, 1, 'pool3', 'VALID', max_pool=False)

        self.dropout1 = self.dropout(self.pool3, 'dropout')

        self.last = self.fc(self.dropout1, self.NUM_CLASSES, 'fc', True, True, True)
#        self.softmax = tf.nn.softmax(self.fc)    # 这一步放在损失函数时用tf.nn.softmax...
