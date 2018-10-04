import chainer
from chainer import links as L
from chainer import functions as F

class Decoder(chainer.ChainList):
    def __init__(self):
        super(Decoder, self).__init__(
            deconv = L.Deconvolution2D(512,512,5,2)
            conv1_1 = L.Convolution2D(512, 512, 3,1,pad=1)
            bn1_1 = L.BatchNormalization(512)
            conv1_2 = L.Convolution2D(512, 512, 3,1,pad=1)
            bn1_2 = L.BatchNormalization(512)
            conv2_1 = L.Convolution2D(512, 512, 3,1,pad=1)
            bn2_1 = L.BatchNormalization(512)
            conv2_2 = L.Convolution2D(512, 512, 3,1,pad=1)
            bn2_2 = L.BatchNormalization(512)
            conv2_3 = L.Convolution2D(512, 512, 3,1,pad=1)
            bn2_3 = L.BatchNormalization(512)
            conv3_1 = L.Convolution2D(256, 256, 3,1,pad=1)
            bn3_1 = L.BatchNormalization(256)
            conv3_2 = L.Convolution2D(256, 256, 3,1,pad=1)
            bn3_2 = L.BatchNormalization(256)
            conv3_3 = L.Convolution2D(256, 256, 3,1,pad=1)
            bn3_3 = L.BatchNormalization(256)
            conv3_4 = L.Convolution2D(256, 256, 3,1,pad=1)
            bn3_4 = L.BatchNormalization(256)
            conv4_1 = L.Convolution2D(256, 256, 3,1,pad=1)
            bn4_1 = L.BatchNormalization(256)
            conv4_2 = L.Convolution2D(256, 256, 3,1,pad=1)
            bn4_2 = L.BatchNormalization(256)
            conv4_3 = L.Convolution2D(256, 256, 3,1,pad=1)
            bn4_3 = L.BatchNormalization(256)
            conv5_1 = L.Convolution2D(128, 128, 3,1,pad=1)
            bn5_1 = L.BatchNormalization(128)
            conv5_2 = L.Convolution2D(128, 128, 3,1,pad=1)
            bn5_2 = L.BatchNormalization(128)
            conv5_3 = L.Convolution2D(128, 128, 3,1,pad=1)
            bn5_3 = L.BatchNormalization(128)
            conv6_1 = L.Convolution2D(64, 64, 3,1,pad=1)
            bn6_1 = L.BatchNormalization(64)
            conv6_2 = L.Convolution2D(64, 64, 3,1,pad=1)
            bn6_2 = L.BatchNormalization(64)
            conv6_3 = L.Convolution2D(64, 64, 3,1,pad=1)
            bn6_3 = L.BatchNormalization(64)
            conv7_1 = L.Convolution2D(32, 32, 3,1,pad=1)
            bn7_1 = L.BatchNormalization(32)
            conv7_2 = L.Convolution2D(32, 32, 3,1,pad=1)
            bn7_2 = L.BatchNormalization(32)
            conv7_3 = L.Convolution2D(32, 32, 3,1,pad=1)
            bn7_3 = L.BatchNormalization(32)
            conv_last = L.Convolution2D(32, 3, 1)
        )
        

    def __call__(self, x):
        x=self.deconv(x)
        x=F.relu(self.bn1_1(self.conv1_1(x)))
        x=F.relu(self.bn1_2(self.conv1_2(x)))
        x=F.resize_images(x,(x.shape[2]*2,x.shape[3]*2))
        x=F.relu(self.bn2_1(self.conv2_1(x)))
        x=F.relu(self.bn2_2(self.conv2_2(x)))
        x=F.relu(self.bn2_3(self.conv2_3(x)))
        x=F.resize_images(x,(x.shape[2]*2,x.shape[3]*2))
        x=F.relu(self.bn3_1(self.conv3_1(x)))
        x=F.relu(self.bn3_2(self.conv3_2(x)))
        x=F.relu(self.bn3_3(self.conv3_3(x)))
        x=F.resize_images(x,(x.shape[2]*2,x.shape[3]*2))
        x=F.relu(self.bn4_1(self.conv4_1(x)))
        x=F.relu(self.bn4_2(self.conv4_2(x)))
        x=F.relu(self.bn4_3(self.conv4_3(x)))
        x=F.resize_images(x,(x.shape[2]*2,x.shape[3]*2))
        x=F.relu(self.bn5_1(self.conv5_1(x)))
        x=F.relu(self.bn5_2(self.conv5_2(x)))
        x=F.relu(self.bn5_3(self.conv5_3(x)))
        x=F.resize_images(x,(x.shape[2]*2,x.shape[3]*2))
        x=F.relu(self.bn6_1(self.conv6_1(x)))
        x=F.relu(self.bn6_2(self.conv6_2(x)))
        x=F.relu(self.bn6_3(self.conv6_3(x)))
        x=F.resize_images(x,(x.shape[2]*2,x.shape[3]*2))
        x=F.relu(self.bn7_1(self.conv7_1(x)))
        x=F.relu(self.bn7_2(self.conv7_2(x)))
        x=F.relu(self.bn7_3(self.conv7_3(x)))
        x=self.conv_last(x)
        return x

