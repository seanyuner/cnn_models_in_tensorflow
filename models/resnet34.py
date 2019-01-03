import tensorflow as tf
from models.base import CNNs


# 定义ResNet34模型
class ResNet34(CNNs):
    def __init__(self, x, num_classes, keep_prob, regularizer=None,
                 write_sum=False, training=True):
        super().__init__(keep_prob, regularizer, write_sum, training)
        self.x = x
        self.NUM_CLASSES = num_classes
        self.create()


    def block(self, base, N, stride, scope):
        with tf.name_scope(scope):
            if stride != 1:
                shortcut = self.conv(base, 1, 1, N, stride, stride,
                                          'shortcut', batch_norm=True, relu=False)
            else:
                shortcut = base

            base = self.conv(base, 3, 3, N, stride, stride, 'conv1', batch_norm=True)
            base = self.conv(base, 3, 3, N, 1, 1, 'conv2', batch_norm=True, relu=False)
            
            base = base + shortcut
            base = tf.nn.relu(base, name='relu')

            return base

        
    def create(self):
        self.conv1 = self.conv(self.x, 7, 7, 64, 2, 2, 'conv1', batch_norm=True)
        self.pool1 = self.pool(self.conv1, 3, 3, 2, 2, 'pool1', 'SAME')

        self.conv2_1 = self.block(self.pool1, 64, 1, 'block2_1')
        self.conv2_2 = self.block(self.conv2_1, 64, 1, 'block2_2')
        self.conv2_3 = self.block(self.conv2_2, 64, 1, 'block2_3')

        self.conv3_1 = self.block(self.conv2_3, 128, 2, 'block3_1')
        self.conv3_2 = self.block(self.conv3_1, 128, 1, 'block3_2')
        self.conv3_3 = self.block(self.conv3_2, 128, 1, 'block3_3')
        self.conv3_4 = self.block(self.conv3_3, 128, 1, 'block3_4')

        self.conv4_1 = self.block(self.conv3_4, 256, 2, 'block4_1')
        self.conv4_2 = self.block(self.conv4_1, 256, 1, 'block4_2')
        self.conv4_3 = self.block(self.conv4_2, 256, 1, 'block4_3')
        self.conv4_4 = self.block(self.conv4_3, 256, 1, 'block4_4')
        self.conv4_5 = self.block(self.conv4_4, 256, 1, 'block4_5')
        self.conv4_6 = self.block(self.conv4_5, 256, 1, 'block4_6')
        
        self.conv5_1 = self.block(self.conv4_6, 512, 2, 'block5_1')
        self.conv5_2 = self.block(self.conv5_1, 512, 1, 'block5_2')
        self.conv5_3 = self.block(self.conv5_2, 512, 1, 'block5_3')

        self.pool2 = self.pool(self.conv5_3, 7, 7, 1, 1, 'pool2', 'VALID', max_pool=False)
        self.last = self.fc(self.pool2, self.NUM_CLASSES, 'fc', True, True, True)
