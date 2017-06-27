# VGG model in Caffe2

The VGG model, described in the technical report [Very Deep Convolutional Networks for Large-Scale Visual Recognition](https://arxiv.org/pdf/1409.1556.pdf), pre-trained on the ImageNet dataset, available on the [VGG website](http://www.robots.ox.ac.uk/~vgg/research/very_deep/).

Both the 16-layer and 19-layer model files can be found in the [model](model/) folder. They require input images of size 224x224. See this [Caffe2 C++ Tutorial](https://github.com/leonardvandriel/caffe2_cpp_tutorial) for an example of how to load this model in Caffe2.

## How the 16-layer model was generated

1. The Caffe model consists of two files:

   - [`VGG_ILSVRC_16_layers.caffemodel`](http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel) containing the model weights for initialization (553 MB)
   - [`VGG_ILSVRC_16_layers_deploy.prototxt`](https://gist.githubusercontent.com/ksimonyan/211839e770f7b538e2d8/raw/0067c9b32f60362c74f4c445a080beed06b07eb3/VGG_ILSVRC_16_layers_deploy.prototxt) containing the model layers for prediction

2. This caffemodel file is of an older protobuf format, requiring some manual translation:

        sed -i '' \
            -e 's/layers {/layer {/g' \
            -e 's/CONVOLUTION/"Convolution"/g' \
            -e 's/RELU/"ReLU"/g' \
            -e 's/POOLING/"Pooling"/g' \
            -e 's/INNER_PRODUCT/"InnerProduct"/g' \
            -e 's/DROPOUT/"Dropout"/g' \
            -e 's/SOFTMAX/"Softmax"/g' \
            VGG_ILSVRC_16_layers_deploy.prototxt

   I'm sure this can be done more efficiently.

3. Now we can run the translator tool that is included in the [Caffe2 sources](https://caffe2.ai/docs/getting-started.html):

        python caffe_translator.py VGG_ILSVRC_16_layers_deploy.prototxt VGG_ILSVRC_16_layers.caffemodel

   This will take a while. On a 2016 MacBook Pro over 4 hours, requiring close to 16GB of memory.

4. This will output two Caffe2 protobuf files:

   - `init_net.pb` containing the model weights for initialization
   - `predict_net.pb` containing the model operators for prediction

