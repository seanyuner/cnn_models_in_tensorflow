from models.base import CNNs


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
        self.conv1 = self.conv(self.x, 11, 11, 96, 4, 4, 'conv1')
        self.lrn1 = self.lrn(self.conv1, 'lrn1', 2, 2.0, 1e-4, 0.75)
        self.pool1 = self.pool(self.lrn1, 3, 3, 2, 2, 'pool1')
        
        #### layer2 ####
        self.conv2 = self.conv(self.pool1, 5, 5, 256, 1, 1, 'conv2')
        self.lrn2 = self.lrn(self.conv2, 'lrn2', 2, 2.0, 1e-4, 0.75)
        self.pool2 = self.pool(self.lrn2, 3, 3, 2, 2, 'pool2')

        #### layer3 ####
        self.conv3 = self.conv(self.pool2, 3, 3, 384, 1, 1, 'conv3')
            
        #### layer4 ####
        self.conv4 = self.conv(self.conv3, 3, 3, 384, 1, 1, 'conv4')
            
        #### layer5 ####
        self.conv5 = self.conv(self.conv4, 3, 3, 256, 1, 1, 'conv5')
        self.pool3 = self.pool(self.conv5, 3, 3, 2, 2, 'pool3')
            
        #### layer6 ####
        self.fc6 = self.fc(self.pool3, 4096, 'fc6', True, True)
        self.dropout1 = self.dropout(self.fc6, 'dropout1')
            
        #### layer7 ####
        self.fc7 = self.fc(self.dropout1, 4096, 'fc7', True)
        self.dropout2 = self.dropout(self.fc7, 'dropout2')
            
        #### layer8 ####
        self.last = self.fc(self.dropout2, self.NUM_CLASSES, 'fc8', False)
