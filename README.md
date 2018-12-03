## cnn_models_in_tensorflow

使用TensorFlow自己搭建一些经典的CNN模型，并使用统一的数据来测试效果。

### 模型：
- AlexNet（[paper](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf))

### 数据：
数据原下载地址在[这里](http://download.tensorflow.org/example_images/flower_photos.tgz)，tgz文件大小约218M，包含5类图片大小不确定的3通道花朵图片，各类数目不等，共计3670张。但由于本人笔记本限制，实验时只取了部分图片，即根目录下datasets中文件，5类各取400张，共2000张。实验时取测试集比例为25%，各100张。

### 文件说明：
- `datasets`：数据集文件夹；
- `images`：一些供ipynb文件调用的可视化或者其他辅助图片；
- `*.ipynb`：实验文档，便于文字记录、图片说明及模型对比等；
- `*.py`:各类CNN模型文件，以模型名命名，如`alexnet.py`是AlexNet模型的实现；
