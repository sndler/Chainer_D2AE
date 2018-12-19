import chainer
from chainer import links as L
from chainer import functions as F

class Decoder(chainer.Chain):
    def __init__(self, config):
        self.config=config
        initializer = chainer.initializers.HeNormal()
        super(Decoder, self).__init__(
            deconv = L.Deconvolution2D(512,512,4,1,0, initialW=initializer),
            conv1_1 = L.Convolution2D(512, 512, 3,1,pad=1, initialW=initializer),
            bn1_1 = L.BatchNormalization(512, 1e-3),
            conv1_2 = L.Convolution2D(512, 512, 3,1,pad=1, initialW=initializer),
            bn1_2 = L.BatchNormalization(512, 1e-3),
            conv2_1 = L.Convolution2D(512, 512, 3,1,pad=1, initialW=initializer),
            bn2_1 = L.BatchNormalization(512, 1e-3),
            conv2_2 = L.Convolution2D(512, 512, 3,1,pad=1, initialW=initializer),
            bn2_2 = L.BatchNormalization(512, 1e-3),
            conv2_3 = L.Convolution2D(512, 512, 3,1,pad=1, initialW=initializer),
            bn2_3 = L.BatchNormalization(512, 1e-3),
            conv3_1 = L.Convolution2D(512, 256, 3,1,pad=1, initialW=initializer),
            bn3_1 = L.BatchNormalization(256, 1e-3),
            conv3_2 = L.Convolution2D(256, 256, 3,1,pad=1, initialW=initializer),
            bn3_2 = L.BatchNormalization(256, 1e-3),
            conv3_3 = L.Convolution2D(256, 256, 3,1,pad=1, initialW=initializer),
            bn3_3 = L.BatchNormalization(256, 1e-3),
            conv3_4 = L.Convolution2D(256, 256, 3,1,pad=1, initialW=initializer),
            bn3_4 = L.BatchNormalization(256, 1e-3),
            conv4_1 = L.Convolution2D(256, 256, 3,1,pad=1, initialW=initializer),
            bn4_1 = L.BatchNormalization(256, 1e-3),
            conv4_2 = L.Convolution2D(256, 256, 3,1,pad=1, initialW=initializer),
            bn4_2 = L.BatchNormalization(256, 1e-3),
            conv4_3 = L.Convolution2D(256, 256, 3,1,pad=1, initialW=initializer),
            bn4_3 = L.BatchNormalization(256, 1e-3),
            conv5_1 = L.Convolution2D(256, 128, 3,1,pad=1, initialW=initializer),
            bn5_1 = L.BatchNormalization(128, 1e-3),
            conv5_2 = L.Convolution2D(128, 128, 3,1,pad=1, initialW=initializer),
            bn5_2 = L.BatchNormalization(128, 1e-3),
            conv5_3 = L.Convolution2D(128, 128, 3,1,pad=1, initialW=initializer),
            bn5_3 = L.BatchNormalization(128, 1e-3),
            conv6_1 = L.Convolution2D(128, 64, 3,1,pad=1,initialW=initializer),
            bn6_1 = L.BatchNormalization(64, 1e-3),
            conv6_2 = L.Convolution2D(64, 64, 3,1,pad=1, initialW=initializer),
            bn6_2 = L.BatchNormalization(64, 1e-3),
            conv6_3 = L.Convolution2D(64, 64, 3,1,pad=1, initialW=initializer),
            bn6_3 = L.BatchNormalization(64, 1e-3),
            conv7_1 = L.Convolution2D(64, 32, 3,1,pad=1, initialW=initializer),
            bn7_1 = L.BatchNormalization(32, 1e-3),
            conv7_2 = L.Convolution2D(32, 32, 3,1,pad=1, initialW=initializer),
            bn7_2 = L.BatchNormalization(32, 1e-3),
            conv7_3 = L.Convolution2D(32, 32, 3,1,pad=1, initialW=initializer),
            bn7_3 = L.BatchNormalization(32, 1e-3),
            conv_last = L.Convolution2D(32, 3, 1, initialW=initializer)
        )
    def mypad(self, x):
        x=F.pad(x,1,mode='reflect')
        x=x[1:-1,1:-1]
        return x
    def __call__(self, x):
        x=self.deconv(x)
        x=F.leaky_relu(self.bn1_1(self.conv1_1(x)))
        x=F.leaky_relu(self.bn1_2(self.conv1_2(x)))
        x=F.resize_images(x,(x.shape[2]*2,x.shape[3]*2))
        x=F.leaky_relu(self.bn2_1(self.conv2_1(x)))
        x=F.leaky_relu(self.bn2_2(self.conv2_2(x)))
        x=F.leaky_relu(self.bn2_3(self.conv2_3(x)))
        x=F.resize_images(x,(x.shape[2]*2,x.shape[3]*2))
        x=F.leaky_relu(self.bn3_1(self.conv3_1(x)))
        x=F.leaky_relu(self.bn3_2(self.conv3_2(x)))
        x=F.leaky_relu(self.bn3_3(self.conv3_3(x)))
        x=F.resize_images(x,(x.shape[2]*2,x.shape[3]*2))
        x=F.leaky_relu(self.bn4_1(self.conv4_1(x)))
        x=F.leaky_relu(self.bn4_2(self.conv4_2(x)))
        x=F.leaky_relu(self.bn4_3(self.conv4_3(x)))
        x=F.resize_images(x,(x.shape[2]*2,x.shape[3]*2))
        x=F.leaky_relu(self.bn5_1(self.conv5_1(x)))
        x=F.leaky_relu(self.bn5_2(self.conv5_2(x)))
        x=F.leaky_relu(self.bn5_3(self.conv5_3(x)))
        x=F.resize_images(x,(x.shape[2]*2,x.shape[3]*2))
        x=F.leaky_relu(self.bn6_1(self.conv6_1(x)))
        x=F.leaky_relu(self.bn6_2(self.conv6_2(x)))
        x=F.leaky_relu(self.bn6_3(self.conv6_3(x)))
        x=F.resize_images(x,(x.shape[2]*2,x.shape[3]*2))
        x=F.leaky_relu(self.bn7_1(self.conv7_1(x)))
        x=F.leaky_relu(self.bn7_2(self.conv7_2(x)))
        x=F.leaky_relu(self.bn7_3(self.conv7_3(x)))
        x=self.conv_last(x)
        #x=F.sigmoid(x)
        x=F.tanh(x)
        return x



