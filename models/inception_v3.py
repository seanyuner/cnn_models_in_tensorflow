import tensorflow as tf
from models.base import CNNs


# 定义inception_v3模型
class Inception_v3(CNNs):
    def __init__(self, x, num_classes, keep_prob, regularizer=None,
                 write_sum=False):
        super().__init__(keep_prob, regularizer, write_sum)
        self.x = x
        self.NUM_CLASSES = num_classes
        self.create()
        
        
    # inception5 module
    def inception5(self, base, N_1x1, N_3x3_reduce, N_3x3, N_d3x3_reduce, N_d3x3_1,
                   N_d3x3_2, N_pool_proj, scope, max_pool=False):
        with tf.name_scope(scope):
            inception_1x1 = self.conv(base, 1, 1, N_1x1, 1, 1, '1x1_1')
            
            inception_3x3_reduce = self.conv(base, 1, 1, N_3x3_reduce, 1, 1, '1x1_2')
            inception_3x3 = self.conv(inception_3x3_reduce, 3, 3, N_3x3, 1, 1, '3x3_1')
            
            inception_d3x3_reduce = self.conv(base, 1, 1, N_d3x3_reduce, 1, 1, '1x1_3')
            inception_d3x3_1 = self.conv(inception_d3x3_reduce, 3, 3, N_d3x3_1, 1, 1, '3x3_2')
            inception_d3x3_2 = self.conv(inception_d3x3_1, 3, 3, N_d3x3_2, 1, 1, '3x3_3')
            
            inception_pool = self.pool(base, 3, 3, 1, 1, 'pool', 'SAME', max_pool)
            inception_pool_proj = self.conv(inception_pool, 1, 1, N_pool_proj, 1, 1, 'pool_proj')
            
            inception_concat = self.concat([inception_1x1, inception_3x3,
                                            inception_d3x3_2, inception_pool_proj], 
                                           3, 'concat')
            return inception_concat
        
        
    # inception6 module
    def inception6(self, base, N_1x1, N_s_reduce, N_s_1, N_s_2, N_d_reduce, N_d_1, N_d_2,
                   N_d_3, N_d_4, N_pool_proj, scope, max_pool=False):
        with tf.name_scope(scope):
            inception_1x1 = self.conv(base, 1, 1, N_1x1, 1, 1, '1x1_1')
            
            inception_s_reduce = self.conv(base, 1, 1, N_s_reduce, 1, 1, '1x1_2')
            inception_s_1 = self.conv(inception_s_reduce, 1, 7, N_s_1, 1, 1, '1x7_1')
            inception_s_2 = self.conv(inception_s_1, 7, 1, N_s_2, 1, 1, '7x1_1')
            
            inception_d_reduce = self.conv(base, 1, 1, N_d_reduce, 1, 1, '1x1_3')
            inception_d_1 = self.conv(inception_d_reduce, 1, 7, N_d_1, 1, 1, '1x7_2')
            inception_d_2 = self.conv(inception_d_1, 7, 1, N_d_2, 1, 1, '7x1_2')
            inception_d_3 = self.conv(inception_d_2, 1, 7, N_d_3, 1, 1, '1x7_3')
            inception_d_4 = self.conv(inception_d_3, 7, 1, N_d_4, 1, 1, '7x1_3')
            
            inception_pool = self.pool(base, 3, 3, 1, 1, 'pool', 'SAME', max_pool)
            inception_pool_proj = self.conv(inception_pool, 1, 1, N_pool_proj, 1, 1, 'pool_proj', 'SAME')
            
            inception_concat = self.concat([inception_1x1, inception_s_2,
                                            inception_d_4, inception_pool_proj], 
                                           3, 'concat')
            return inception_concat
        
        
    # inception7 module
    def inception7(self, base, N_1x1, N_s_reduce, N_s_1, N_s_2, N_d_reduce, N_d_1,
                   N_d_2_1, N_d_2_2, N_pool_proj, scope, max_pool=False):
        with tf.name_scope(scope):
            inception_1x1 = self.conv(base, 1, 1, N_1x1, 1, 1, '1x1_1')
            
            inception_s_reduce = self.conv(base, 1, 1, N_s_reduce, 1, 1, '1x1_2')
            inception_s_1 = self.conv(inception_s_reduce, 1, 3, N_s_1, 1, 1, '1x3_1')
            inception_s_2 = self.conv(inception_s_reduce, 3, 1, N_s_2, 1, 1, '3x1_1')
            
            inception_d_reduce = self.conv(base, 1, 1, N_d_reduce, 1, 1, '1x1_3')
            inception_d_1 = self.conv(inception_d_reduce, 3, 3, N_d_1, 1, 1, '3x3')
            inception_d_2_1 = self.conv(inception_d_1, 1, 3, N_d_2_1, 1, 1, '1x3_2')
            inception_d_2_2 = self.conv(inception_d_1, 3, 1, N_d_2_2, 1, 1, '3x1_2')
            
            inception_pool = self.pool(base, 3, 3, 1, 1, 'pool', 'SAME', max_pool)
            inception_pool_proj = self.conv(inception_pool, 1, 1, N_pool_proj, 1, 1, 'pool_proj')
            
            inception_concat = self.concat([inception_1x1, inception_s_2,
                                            inception_d_2_1, inception_d_2_2, 
                                            inception_pool_proj], 
                                           3, 'concat')
            return inception_concat
        
    
    def reduction6(self, base, scope, max_pool=True):
        with tf.name_scope(scope):
            inception_3x3 = self.conv(base, 3, 3, 384, 2, 2, '3x3_1', 'VALID')
            
            inception_d_reduce = self.conv(base, 1, 1, 64, 1, 1, '1x1')
            inception_d_1 = self.conv(inception_d_reduce, 3, 3, 96, 1, 1, '3x3_2')
            inception_d_2 = self.conv(inception_d_1, 3, 3, 96, 2, 2, '3x3_3', 'VALID')
            
            inception_pool = self.pool(base, 3, 3, 2, 2, 'pool', 'VALID', max_pool)
            
            inception_concat = self.concat([inception_3x3, 
                                            inception_d_2, 
                                            inception_pool], 
                                           3, 'concat')
            return inception_concat
        
        
    def reduction7(self, base, scope, max_pool=True):
        with tf.name_scope(scope):
            inception_3x3_reduce = self.conv(base, 1, 1, 192, 1, 1, '1x1_1')
            inception_3x3 = self.conv(inception_3x3_reduce, 3, 3, 320, 2, 2, '3x3_1', 'VALID')
            
            inception_frac_reduce = self.conv(base, 1, 1, 192, 1, 1, '1x1_2')
            inception_frac_1 = self.conv(inception_frac_reduce, 1, 7, 192, 1, 1, '1x7')
            inception_frac_2 = self.conv(inception_frac_1, 7, 1, 192, 1, 1, '7x1')
            inception_frac_3 = self.conv(inception_frac_2, 3, 3, 192, 2, 2, '3x3_2', 'VALID')
            
            inception_pool = self.pool(base, 3, 3, 2, 2, 'pool', 'VALID', max_pool)
            
            inception_concat = self.concat([inception_3x3, 
                                            inception_frac_3, 
                                            inception_pool], 
                                           3, 'concat')
            return inception_concat
        
        
    def create(self):
        self.conv1 = self.conv(self.x, 3, 3, 32, 2, 2, 'conv1', 'VALID')
        self.conv2 = self.conv(self.conv1, 3, 3, 32, 1, 1, 'conv2', 'VALID')
        self.conv3 = self.conv(self.conv2, 3, 3, 64, 1, 1, 'conv3')
        self.pool1 = self.pool(self.conv3, 3, 3, 2, 2, 'pool1')
        
        self.conv4 = self.conv(self.pool1, 3, 3, 80, 1, 1, 'conv4', 'VALID')
        self.conv5 = self.conv(self.conv4, 3, 3, 192, 2, 2, 'conv5', 'VALID')
        self.conv6 = self.conv(self.conv5, 3, 3, 288, 1, 1, 'conv6')
        
        self.inception5a_concat = self.inception5(self.conv6, 64, 48, 64, 64, 96, 96, 64, 'inception5a')
        self.inception5b_concat = self.inception5(self.inception5a_concat, 64, 48, 64, 64, 96, 96, 64, 'inception5b')
        self.inception5c_concat = self.inception5(self.inception5b_concat, 64, 48, 64, 64, 96, 96, 64, 'inception5c')
        
        self.inception6a_concat = self.reduction6(self.inception5c_concat, 'inception6a')
        self.inception6b_concat = self.inception6(self.inception6a_concat, 192, 128, 128, 192, 128, 128, 128, 128, 192, 192, 'inception6b')
        self.inception6c_concat = self.inception6(self.inception6b_concat, 192, 128, 128, 192, 128, 128, 128, 128, 192, 192, 'inception6c')
        self.inception6d_concat = self.inception6(self.inception6c_concat, 192, 128, 128, 192, 128, 128, 128, 128, 192, 192, 'inception6d')
        self.inception6e_concat = self.inception6(self.inception6d_concat, 192, 128, 128, 192, 128, 128, 128, 128, 192, 192, 'inception6e')
        
        self.inception7a_concat = self.reduction7(self.inception6e_concat, 'inception7a')
        self.inception7b_concat = self.inception7(self.inception7a_concat, 320, 384, 384, 384, 448, 384, 384, 384, 192, 'inception7b')
        self.inception7c_concat = self.inception7(self.inception7b_concat, 320, 384, 384, 384, 448, 384, 384, 384, 192, 'inception7c')
        
        self.pool2 = self.pool(self.inception7c_concat, 8, 8, 1, 1, 'pool2', max_pool=False)
        self.dropout1 = self.dropout(self.pool2, 'dropout')
        self.last = self.fc(self.dropout1, self.NUM_CLASSES, 'fc', True, True)
#        self.softmax = tf.nn.softmax(self.fc)    # 这一步放在损失函数时用tf.nn.softmax...
