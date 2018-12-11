import tensorflow as tf
from models.base import CNNs


# 定义GoogLeNet(inception_v1)模型
class GoogLeNet(CNNs):
    def __init__(self, x, num_classes, keep_prob, regularizer=None,
                 write_sum=False, auxil=None):
        super().__init__(keep_prob, regularizer, write_sum)
        self.x = x
        self.NUM_CLASSES = num_classes
        self.AUXIL = auxil
        self.create()
        

    # inception module
    def inception(self, base, N_1x1, N_3x3_reduce, N_3x3, N_5x5_reduce, N_5x5, N_pool_proj, scope):
        with tf.name_scope(scope):
            
            inception_1x1 = self.conv(base, 1, 1, N_1x1, 1, 1, '1x1')
            
            inception_3x3_reduce = self.conv(base, 1, 1, N_3x3_reduce,  1, 1, '3x3_reduce')
            inception_3x3 = self.conv(inception_3x3_reduce, 3, 3, N_3x3, 1, 1, '3x3')
            
            inception_5x5_reduce = self.conv(base, 1, 1, N_5x5_reduce, 1, 1, '5x5_reduce')
            inception_5x5 = self.conv(inception_5x5_reduce, 5, 5, N_5x5, 1, 1, '5x5')
            
            inception_pool = self.pool(base, 3, 3, 1, 1, 'pool', 'SAME')
            inception_pool_proj = self.conv(inception_pool, 1, 1, N_pool_proj, 1, 1, 'pool_proj')
            
            inception_concat = self.concat([inception_1x1, inception_3x3,
                                            inception_5x5, inception_pool_proj],
                                           3, 'concat')
            return inception_concat


    # auxiliary classifier
    def auxil(self, base, scope):
        with tf.name_scope(scope):
            auxil_pool = self.pool(base, 5, 5, 3, 3, 'pool', max_pool=False)
            auxil_conv = self.conv(auxil_pool, 1, 1, 128, 1, 1, 'conv')
            auxil_fc1 = self.fc(auxil_conv, 1024, 'fc1', True, True)
            auxil_dropout = self.dropout(auxil_fc1, 'dropout')
            auxil_fc2 = self.fc(auxil_dropout, self.NUM_CLASSES, 'fc2', True)
            return auxil_fc2
        
        
    def create(self):
        self.conv1 = self.conv(self.x, 7, 7, 64, 2, 2, 'conv1')
        self.pool1 = self.pool(self.conv1, 3, 3, 2, 2, 'pool1', 'SAME')
        self.lrn1 = self.lrn(self.pool1, 'lrn1', 2, 2.0, 1e-4, 0.75)

        self.conv2_reduce = self.conv(self.lrn1, 1, 1, 64, 1, 1, 'conv2_reduce', 'VALID')
        self.conv2 = self.conv(self.conv2_reduce, 3, 3, 192, 1, 1, 'conv2')
        self.lrn2 = self.lrn(self.conv2, 'lrn2', 2, 2.0, 1e-4, 0.75)
        self.pool2 = self.pool(self.lrn2, 3, 3, 2, 2, 'pool2', 'SAME')
        
        self.inception3a_concat = self.inception(self.pool2, 64, 96, 128, 16, 32, 32, 'inception3a')
        self.inception3b_concat = self.inception(self.inception3a_concat, 64, 96, 128, 16, 32, 32, 'inception3b')
        
        self.pool3 = self.pool(self.inception3b_concat, 3, 3, 2, 2, 'pool3', 'SAME')
        
        self.inception4a_concat = self.inception(self.pool3, 192, 96, 208, 16, 48, 64, 'inception4a')
        self.inception4b_concat = self.inception(self.inception4a_concat, 160, 112, 224, 24, 64, 64, 'inception4b')
        self.inception4c_concat = self.inception(self.inception4b_concat, 128, 128, 256, 24, 64, 64, 'inception4c')
        self.inception4d_concat = self.inception(self.inception4c_concat, 112, 144, 288, 32, 64, 64, 'inception4d')
        self.inception4e_concat = self.inception(self.inception4d_concat, 256, 160, 320, 32, 128, 128, 'inception4e')
        
        self.pool4 = self.pool(self.inception4e_concat, 3, 3, 2, 2, 'pool4', 'SAME')
        
        self.inception5a_concat = self.inception(self.pool4, 256, 160, 320, 32, 128, 128, 'inception5a')
        self.inception5b_concat = self.inception(self.inception5a_concat, 384, 192, 384, 48, 128, 128, 'inception5b')
        
        self.pool5 = self.pool(self.inception5b_concat, 7, 7, 1, 1, 'pool5', 'VALID', max_pool=False)

        self.dropout1 = self.dropout(self.pool5, 'dropout')

        self.last = self.fc(self.dropout1, self.NUM_CLASSES, 'fc', True, True)
#        self.softmax = tf.nn.softmax(self.fc)    # 这一步放在损失函数时用tf.nn.softmax...（下同）

        if self.AUXIL is not None:
            self.auxil1_last = self.auxil(self.inception4a_concat, 'auxil1')
            self.auxil2_last = self.auxil(self.inception4d_concat, 'auxil2')
            self.last = tf.reduce_mean([self.last, 0.3 * self.auxil1_last, 0.3 * self.auxil2_last], axis=0)
