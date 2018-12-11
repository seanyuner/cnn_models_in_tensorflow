## cnn_models_in_tensorflow

使用TensorFlow自己搭建一些经典的CNN模型，并使用统一的数据来测试效果。

### 模型：
- AlexNet, 2012 ([paper](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks))
- GoogLeNet, 2014 ([paper](https://arxiv.org/abs/1409.4842))
- VGG, 2014 ([paper](https://arxiv.org/abs/1409.1556))
- Inception_v2, ([paper](https://arxiv.org/abs/1502.03167))
- Inception_v3, ([paper](https://arxiv.org/abs/1512.00567))

### 数据：
数据原下载地址在[这里](http://download.tensorflow.org/example_images/flower_photos.tgz)，tgz文件大小约218M，包含5类图片大小不确定的3通道花朵图片，各类数目不等，共计3670张。但由于本人笔记本限制，实验时只取了部分图片，即根目录下datasets中文件，5类各取400张，共2000张。实验时取测试集比例为25%，各100张。

### 文件说明：
- `datasets`：数据集文件夹；
- `images`：一些供ipynb文件调用的可视化或者其他辅助图片文件夹；
- `models`：搭建模型文件夹；
- `*.ipynb`：实验记录文档，一些文字记录、图片说明，包括迭代记录、模型说明和对比等；
- `alexnet.py`：最开始的第一个模型，此处意义不大，在models文件夹中也有；
