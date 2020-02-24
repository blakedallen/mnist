# mnist
Solving MNIST the hello world of visual classification tasks

## Usage

```
python3 main.py
```

## Overview

An example convolutional neural network solving mnist

Architecture overview:
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d_1 (Conv2D)            (None, 26, 26, 32)        320
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 13, 13, 32)        0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 11, 11, 64)        18496
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 5, 5, 64)          0
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 3, 3, 128)         73856
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 1, 1, 128)         0
_________________________________________________________________
flatten_1 (Flatten)          (None, 128)               0
_________________________________________________________________
dropout_1 (Dropout)          (None, 128)               0
_________________________________________________________________
dropout_2 (Dropout)          (None, 128)               0
_________________________________________________________________
dense_1 (Dense)              (None, 10)                1290
=================================================================
Total params: 93,962
Trainable params: 93,962
Non-trainable params: 0

```

## Thought Process

This is a simple deep convolutional network. It gets 98% accuracy after 12 epochs. 
Convolutional layers power the inference of the 2d images. 
Maxpooling to downsample the detection of features in feature maps. 
Flatten layer to convert the shape from 2d matrices to vectors.
Dropout layers to prevent overfitting.
Dense layer for the classification task. 
