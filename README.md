# Pre-trained Caffe2 Models

*A collection of common deep learning models, pre-trained for Caffe2*

## VGG

The VGG model, described in the technical report [Very Deep Convolutional Networks for Large-Scale Visual Recognition](https://arxiv.org/pdf/1409.1556.pdf), pre-trained on the ImageNet dataset, available on the [VGG website](http://www.robots.ox.ac.uk/~vgg/research/very_deep/).

Both the 16-layer and 19-layer model files can be found in the [model](model/) folder. They require input images of size 224x224. See this [Caffe2 C++ Tutorial](https://github.com/leonardvandriel/caffe2_cpp_tutorial) for an example of how to load this model in Caffe2.

### How the 16-layer model was generated

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

## ResNet

The ResNet model, described in the technical report [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf), pre-trained on the ImageNet dataset, available on the [KaimingHe GitHub](https://github.com/KaimingHe/deep-residual-networks).

The three original models (ResNet-50, ResNet-101, and ResNet-152) can be found in the [model](model/) folder. They require input images of size 224x224. See this [Caffe2 C++ Tutorial](https://github.com/leonardvandriel/caffe2_cpp_tutorial) for an example of how to load this model in Caffe2.

### How the ResNet-101 model was generated

1. The Caffe model consists of two files:

   - `ResNet-101-deploy.prototxt` containing the model weights for initialization (170 MB)
   - `ResNet-101-model.caffemodel` containing the model layers for prediction

2. Now we can run the translator tool that is included in the [Caffe2 sources](https://caffe2.ai/docs/getting-started.html):

        python caffe_translator.py ResNet-101-deploy.prototxt ResNet-101-model.caffemodel

   This will take a while. On a 2016 MacBook Pro over 30 minutes, requiring close to 4GB of memory.

3. There seems to be two bugs in the translator:

    - The `SpatialBN` operator has equal input and output, which is not allowed for this operator (`_unique`).
    - Some `_w` and `_b` blobs have the same name, resulting in predictable collisions (`_second`).

    These problems are resolved by converting the model using the following C++ code:

        NetDef init_net, predict_net;

        CAFFE_ENFORCE(ReadProtoFromFile("resnet50_init_net.pb", &init_net));
        CAFFE_ENFORCE(ReadProtoFromFile("resnet50_predict_net.pb", &predict_net));

        // remove duplicate initializations
        std::set<string> existing;
        for (const OperatorDef &constop: init_net.op()) {
          OperatorDef &op = const_cast<OperatorDef &>(constop);
          if (existing.find(op.output(0)) != existing.end()) {
            int i = 0, j = 0;
            for (auto &external: predict_net.external_input()) {
              if (external == op.output(0)) {
                j++;
                if (j == 2) {
                  predict_net.set_external_input(i, op.output(0) + "_second");
                  break;
                }
              }
              i++;
            }
            op.set_output(0, op.output(0) + "_second");
          }
          existing.insert(op.output(0));
        }

        // fix self-reference in SpatialBN operations
        string name;
        for (const OperatorDef &constop: predict_net.op()) {
          OperatorDef &op = const_cast<OperatorDef &>(constop);
          if (name.size() && op.input(0) == name) {
            op.set_input(0, name + "_unique");
            if (existing.find(op.input(1) + "_second") != existing.end()) {
              op.set_input(1, op.input(1) + "_second");
            }
          }
          if (op.output(0) == name) {
            name = "";
          }
          if (op.type() == "SpatialBN" && op.input(0) == op.output(0)) {
            name = op.input(0);
            op.set_output(0, name + "_unique");
          }
        }

        WriteProtoToBinaryFile(init_net, "init_net.pb");
        WriteProtoToBinaryFile(predict_net, "predict_net.pb");

4. This will output two Caffe2 protobuf files:

   - `init_net.pb` containing the model weights for initialization
   - `predict_net.pb` containing the model operators for prediction
