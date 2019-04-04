import chainer
from chainer import links as L
from chainer import functions as F
from chainer.functions.activation.relu import ReLU
#from chainer import Sequential

class BasicConv2d(chainer.Chain):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__(
            conv = L.Convolution2D(in_planes, out_planes, ksize=kernel_size,stride=stride,pad=padding,nobias=True),
            bn = L.BatchNormalization(out_planes,
                                    eps=0.001, # value found in tensorflow
                                    )
        )

    def __call__(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        return x

"""
class Sequential(chainer.ChainList):
    def __init__(self, blocks):
        super(Sequential, self).__init__()
        layers = []
        for i in range(len(blocks)):
            layers.append(blocks[i])
        super().__init__(*layers)
    def __call__(self, x):
        for i in range(len(self)):
            x = self[i](x)
        return x
"""

class Mixed_5b(chainer.Chain):

    def __init__(self):
        super(Mixed_5b, self).__init__()
        with self.init_scope():
            self.branch0 = BasicConv2d(192, 96, kernel_size=1, stride=1)

            self.branch1_0 = BasicConv2d(192, 48, kernel_size=1, stride=1)
            self.branch1_1 = BasicConv2d(48, 64, kernel_size=5, stride=1, padding=2)

            self.branch2_0 = BasicConv2d(192, 64, kernel_size=1, stride=1)
            self.branch2_1 = BasicConv2d(64, 96, kernel_size=3, stride=1, padding=1)
            self.branch2_2 = BasicConv2d(96, 96, kernel_size=3, stride=1, padding=1)

            #self.branch3_0 = AveragePooling2D(3, 1, 1, False)
            self.branch3_1 = BasicConv2d(192, 64, kernel_size=1, stride=1)

    def __call__(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1_0(x)
        x1 = self.branch1_1(x1)
        x2 = self.branch2_0(x)
        x2 = self.branch2_1(x2)
        x2 = self.branch2_2(x2)
        #x3 = self.branch3_0(x)
        x3 = F.average_pooling_2d(x, ksize=3, stride=1, pad=1)
        x3 = self.branch3_1(x3)
        #out = torch.cat((x0, x1, x2, x3), 1)
        out = F.concat((x0, x1, x2, x3), 1)
        return out


class Block35(chainer.Chain):

    def __init__(self, scale=1.0):
        super(Block35, self).__init__()
        with self.init_scope():
            self.scale = scale
            self.branch0 = BasicConv2d(320, 32, kernel_size=1, stride=1)

            self.branch1_0 = BasicConv2d(320, 32, kernel_size=1, stride=1)
            self.branch1_1 = BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)

            self.branch2_0 = BasicConv2d(320, 32, kernel_size=1, stride=1)
            self.branch2_1 = BasicConv2d(32, 48, kernel_size=3, stride=1, padding=1)
            self.branch2_2 = BasicConv2d(48, 64, kernel_size=3, stride=1, padding=1)

            self.conv2d = L.Convolution2D(128, 320, ksize=1,stride=1,pad=0)
            #self.relu = ReLU()

    def __call__(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1_0(x)
        x1 = self.branch1_1(x1)
        x2 = self.branch2_0(x)
        x2 = self.branch2_1(x2)
        x2 = self.branch2_2(x2)
        #out = torch.cat((x0, x1, x2), 1)
        out = F.concat((x0, x1, x2), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = F.relu(out)
        return out


class Mixed_6a(chainer.Chain):

    def __init__(self):
        super(Mixed_6a, self).__init__()
        with self.init_scope():
            self.branch0 = BasicConv2d(320, 384, kernel_size=3, stride=2)

            self.branch1_0 = BasicConv2d(320, 256, kernel_size=1, stride=1)
            self.branch1_1 = BasicConv2d(256, 256, kernel_size=3, stride=1, padding=1)
            self.branch1_2 = BasicConv2d(256, 384, kernel_size=3, stride=2)

            #self.branch2 = MaxPooling2D(3, 2, 0, True)

    def __call__(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1_0(x)
        x1 = self.branch1_1(x1)
        x1 = self.branch1_2(x1)
        #x2 = self.branch2(x)
        x2 = F.max_pooling_2d(x, ksize=(3, 3),stride=2, pad=0)

        #out = torch.cat((x0, x1, x2), 1)
        out = F.concat((x0, x1, x2), 1)
        return out


class Block17(chainer.Chain):

    def __init__(self, scale=1.0):
        super(Block17, self).__init__()
        with self.init_scope():
            self.scale = scale

            self.branch0 = BasicConv2d(1088, 192, kernel_size=1, stride=1)

            self.branch1_0 = BasicConv2d(1088, 128, kernel_size=1, stride=1)
            self.branch1_1 = BasicConv2d(128, 160, kernel_size=(1,7), stride=1, padding=(0,3))
            self.branch1_2 = BasicConv2d(160, 192, kernel_size=(7,1), stride=1, padding=(3,0))

            self.conv2d = L.Convolution2D(384, 1088, ksize=1, stride=1, pad=0)
            #self.relu = ReLU()

    def __call__(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1_0(x)
        x1 = self.branch1_1(x1)
        x1 = self.branch1_2(x1)
        #out = torch.cat((x0, x1), 1)
        out = F.concat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = F.relu(out)
        return out


class Mixed_7a(chainer.Chain):

    def __init__(self):
        super(Mixed_7a, self).__init__()
        with self.init_scope():
            self.branch0_0 = BasicConv2d(1088, 256, kernel_size=1, stride=1)
            self.branch0_1 = BasicConv2d(256, 384, kernel_size=3, stride=2)

            self.branch1_0 = BasicConv2d(1088, 256, kernel_size=1, stride=1)
            self.branch1_1 = BasicConv2d(256, 288, kernel_size=3, stride=2)

            self.branch2_0 = BasicConv2d(1088, 256, kernel_size=1, stride=1)
            self.branch2_1 = BasicConv2d(256, 288, kernel_size=3, stride=1, padding=1)
            self.branch2_2 = BasicConv2d(288, 320, kernel_size=3, stride=2)

            #self.branch3 = MaxPooling2D(3, 2, 0, True)

    def __call__(self, x):
        x0 = self.branch0_0(x)
        x0 = self.branch0_1(x0)
        x1 = self.branch1_0(x)
        x1 = self.branch1_1(x1)
        x2 = self.branch2_0(x)
        x2 = self.branch2_1(x2)
        x2 = self.branch2_2(x2)
        #x3 = self.branch3(x)
        x3 = F.max_pooling_2d(x,ksize=3,stride=2)
        #out = torch.cat((x0, x1, x2, x3), 1)
        out = F.concat((x0, x1, x2, x3), 1)
        return out


class Block8(chainer.Chain):

    def __init__(self, scale=1.0, noReLU=False):
        super(Block8, self).__init__()
        with self.init_scope():
            self.scale = scale
            self.noReLU = noReLU
            self.branch0 = BasicConv2d(2080, 192, kernel_size=1, stride=1)
            self.branch1_0 = BasicConv2d(2080, 192, kernel_size=1, stride=1)
            self.branch1_1 = BasicConv2d(192, 224, kernel_size=(1,3), stride=1, padding=(0,1))
            self.branch1_2 = BasicConv2d(224, 256, kernel_size=(3,1), stride=1, padding=(1,0))
            self.conv2d = L.Convolution2D(448, 2080, ksize=1, stride=1, pad=0)
            #if not self.noReLU:
            #    self.relu = L.ReLU(inplace=False)

    def __call__(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1_0(x)
        x1 = self.branch1_1(x1)
        x1 = self.branch1_2(x1)
        #out = torch.cat((x0, x1), 1)
        out = F.concat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        if not self.noReLU:
            out = F.relu(out)
        return out


class InceptionResNetV2(chainer.Chain):
    def __init__(self, num_classes=1001):
        super(InceptionResNetV2, self).__init__()
        # Special attributs
        self.input_space = None
        self.input_size = (299, 299, 3)
        self.mean = None
        self.std = None
        with self.init_scope():
            # Modules
            self.conv2d_1a = BasicConv2d(3, 32, kernel_size=3, stride=2)
            self.conv2d_2a = BasicConv2d(32, 32, kernel_size=3, stride=1)
            self.conv2d_2b = BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1)
            #self.maxpool_3a = MaxPooling2D(3, 2, 0, True)
            self.conv2d_3b = BasicConv2d(64, 80, kernel_size=1, stride=1)
            self.conv2d_4a = BasicConv2d(80, 192, kernel_size=3, stride=1)
            #self.maxpool_5a = MaxPooling2D(3, 2, 0, True)
            self.mixed_5b = Mixed_5b()
            self.repeat_0_0 = Block35(scale=0.17)
            self.repeat_0_1 = Block35(scale=0.17)
            self.repeat_0_2 = Block35(scale=0.17)
            self.repeat_0_3 = Block35(scale=0.17)
            self.repeat_0_4 = Block35(scale=0.17)
            self.repeat_0_5 = Block35(scale=0.17)
            self.repeat_0_6 = Block35(scale=0.17)
            self.repeat_0_7 = Block35(scale=0.17)
            self.repeat_0_8 = Block35(scale=0.17)
            self.repeat_0_9 = Block35(scale=0.17)

            self.mixed_6a = Mixed_6a()
            self.repeat_1_0 = Block17(scale=0.10)
            self.repeat_1_1 = Block17(scale=0.10)
            self.repeat_1_2 = Block17(scale=0.10)
            self.repeat_1_3 = Block17(scale=0.10)
            self.repeat_1_4 = Block17(scale=0.10)
            self.repeat_1_5 = Block17(scale=0.10)
            self.repeat_1_6 = Block17(scale=0.10)
            self.repeat_1_7 = Block17(scale=0.10)
            self.repeat_1_8 = Block17(scale=0.10)
            self.repeat_1_9 = Block17(scale=0.10)
            self.repeat_1_10 = Block17(scale=0.10)
            self.repeat_1_11 = Block17(scale=0.10)
            self.repeat_1_12 = Block17(scale=0.10)
            self.repeat_1_13 = Block17(scale=0.10)
            self.repeat_1_14 = Block17(scale=0.10)
            self.repeat_1_15 = Block17(scale=0.10)
            self.repeat_1_16 = Block17(scale=0.10)
            self.repeat_1_17 = Block17(scale=0.10)
            self.repeat_1_18 = Block17(scale=0.10)
            self.repeat_1_19 = Block17(scale=0.10)

            self.mixed_7a = Mixed_7a()
            self.repeat_2_0 = Block8(scale=0.20)
            self.repeat_2_1 = Block8(scale=0.20)
            self.repeat_2_2 = Block8(scale=0.20)
            self.repeat_2_3 = Block8(scale=0.20)
            self.repeat_2_4 = Block8(scale=0.20)
            self.repeat_2_5 = Block8(scale=0.20)
            self.repeat_2_6 = Block8(scale=0.20)
            self.repeat_2_7 = Block8(scale=0.20)
            self.repeat_2_8 = Block8(scale=0.20)

            self.block8 = Block8(noReLU=True)
            self.conv2d_7b = BasicConv2d(2080, 1536, kernel_size=1, stride=1)
            #self.avgpool_1a = AveragePooling2D(8, 1, 0, False)
            #self.last_linear = nn.Linear(1536, num_classes)
            self.last_linear = L.Linear(None, num_classes)

    def features(self, input):
        x = self.conv2d_1a(input)
        x = self.conv2d_2a(x)
        x = self.conv2d_2b(x)
        #x = self.maxpool_3a(x)
        x = F.max_pooling_2d(x,ksize=(3,3),stride=2)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        #x = self.maxpool_5a(x)
        x = F.max_pooling_2d(x,ksize=(3,3),stride=2)
        x = self.mixed_5b(x)
        x = self.repeat_0_0(x)
        x = self.repeat_0_1(x)
        x = self.repeat_0_2(x)
        x = self.repeat_0_3(x)
        x = self.repeat_0_4(x)
        x = self.repeat_0_5(x)
        x = self.repeat_0_6(x)
        x = self.repeat_0_7(x)
        x = self.repeat_0_8(x)
        x = self.repeat_0_9(x)
        x = self.mixed_6a(x)
        x = self.repeat_1_0(x)
        x = self.repeat_1_1(x)
        x = self.repeat_1_2(x)
        x = self.repeat_1_3(x)
        x = self.repeat_1_4(x)
        x = self.repeat_1_5(x)
        x = self.repeat_1_6(x)
        x = self.repeat_1_7(x)
        x = self.repeat_1_8(x)
        x = self.repeat_1_9(x)
        x = self.repeat_1_10(x)
        x = self.repeat_1_11(x)
        x = self.repeat_1_12(x)
        x = self.repeat_1_13(x)
        x = self.repeat_1_14(x)
        x = self.repeat_1_15(x)
        x = self.repeat_1_16(x)
        x = self.repeat_1_17(x)
        x = self.repeat_1_18(x)
        x = self.repeat_1_19(x)
        x = self.mixed_7a(x)
        x = self.repeat_2_0(x)
        x = self.repeat_2_1(x)
        x = self.repeat_2_2(x)
        x = self.repeat_2_3(x)
        x = self.repeat_2_4(x)
        x = self.repeat_2_5(x)
        x = self.repeat_2_6(x)
        x = self.repeat_2_7(x)
        x = self.repeat_2_8(x)
        x = self.block8(x)
        x = self.conv2d_7b(x)
        return x

    def logits(self, features):
        #x = self.avgpool_1a(features)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        return x

    def __call__(self, input):
        x = self.features(input)
        #x = self.logits(x)
        return x

def inceptionresnetv2(num_classes=1000, pretrained='imagenet'):
    r"""InceptionResNetV2 model architecture from the
    `"InceptionV4, Inception-ResNet..." <https://arxiv.org/abs/1602.07261>`_ paper.
    """
    model = InceptionResNetV2(num_classes=num_classes)
    return model
