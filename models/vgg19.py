from models.base import CNNs


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
        self.conv1_1 = self.conv(self.x, 3, 3, 64, 1, 1, 'b1_conv1')
        self.conv1_2 = self.conv(self.conv1_1, 3, 3, 64, 1, 1, 'b1_conv2')
        self.pool1_1 = self.pool(self.conv1_2, 2, 2, 2, 2, 'b1_pool1', 'SAME')
            
        #### block2 ####
        self.conv2_1 = self.conv(self.pool1_1, 3, 3, 128, 1, 1, 'b2_conv1')
        self.conv2_2 = self.conv(self.conv2_1, 3, 3, 128, 1, 1, 'b2_conv2')
        self.pool2_1 = self.pool(self.conv2_2, 2, 2, 2, 2, 'b2_pool1', 'SAME')
            
        #### block3 ####
        self.conv3_1 = self.conv(self.pool2_1, 3, 3, 256, 1, 1, 'b3_conv1')
        self.conv3_2 = self.conv(self.conv3_1, 3, 3, 256, 1, 1, 'b3_conv2')
        self.conv3_3 = self.conv(self.conv3_2, 3, 3, 256, 1, 1, 'b3_conv3')
        self.conv3_4 = self.conv(self.conv3_3, 3, 3, 256, 1, 1, 'b3_conv4')
        self.pool3_1 = self.pool(self.conv3_4, 2, 2, 2, 2, 'b3_pool1', 'SAME')
        
        #### block4 ####
        self.conv4_1 = self.conv(self.pool3_1, 3, 3, 512, 1, 1, 'b4_conv1')
        self.conv4_2 = self.conv(self.conv4_1, 3, 3, 512, 1, 1, 'b4_conv2')
        self.conv4_3 = self.conv(self.conv4_2, 3, 3, 512, 1, 1, 'b4_conv3')
        self.conv4_4 = self.conv(self.conv4_3, 3, 3, 512, 1, 1, 'b4_conv4')
        self.pool4_1 = self.pool(self.conv4_4, 2, 2, 2, 2, 'b4_pool1', 'SAME')
            
        #### block5 ####
        self.conv5_1 = self.conv(self.pool4_1, 3, 3, 512, 1, 1, 'b5_conv1')
        self.conv5_2 = self.conv(self.conv5_1, 3, 3, 512, 1, 1, 'b5_conv2')
        self.conv5_3 = self.conv(self.conv5_2, 3, 3, 512, 1, 1, 'b5_conv3')
        self.conv5_4 = self.conv(self.conv5_3, 3, 3, 512, 1, 1, 'b5_conv4')
        self.pool5_1 = self.pool(self.conv5_4, 2, 2, 2, 2,  'b5_pool1', 'SAME')
            
        #### fc ####
        self.fc1 = self.fc(self.pool5_1, 4096, 'fc_fc1', True, True)
        self.fc2 = self.fc(self.fc1, 4096, 'fc_fc2', True)
        self.last = self.fc(self.fc2, self.NUM_CLASSES, 'fc_fc3', True)
#       self.softmax = tf.nn.softmax(self.fc3)    # 这一步放在损失函数时用tf.nn.softmax...
