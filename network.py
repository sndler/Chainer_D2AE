
import chainer
from chainer import links as L
from chainer import functions as F
import numpy as np
import cupy as xp
import cupy
#import encoder.Encoder
#import decoder.Decoder

from chainer import Variable
import chainer.cuda

class Subnet(chainer.Chain):
    def __init__(self):
        initializer = chainer.initializers.HeNormal()
        super(Subnet, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(1536, 256, 3,1,pad=1, initialW=initializer)
            self.bn1 = L.BatchNormalization(256, 1e-3)
            self.conv2 = L.Convolution2D(256, 256, 3,1,pad=1, initialW=initializer)
            self.bn2 = L.BatchNormalization(256, 1e-3)
            self.conv3 = L.Convolution2D(256, 256, 3,1,pad=1, initialW=initializer)
            self.bn3 = L.BatchNormalization(256, 1e-3)
            self.fc = L.Convolution2D(256*6*6, 256,1,1,pad=0, initialW=initializer)
            #self.fc = L.Convolution2D(256, 256,1,1,pad=0, initialW=initializer)
            self.bn4 = L.BatchNormalization(256, 1e-3)

    def __call__(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        #x = F.average_pooling_2d(x, (x.shape[2],x.shape[3]))
        x = F.reshape(x,(x.shape[0],256*6*6,1,1))
        x = F.relu(self.bn4(self.fc(x)))
        #x = F.reshape(x, (x.shape[0],x.shape[1]))
        return x

class Identity(chainer.Chain):
    def __init__(self, identityn):
        initializer = chainer.initializers.HeNormal()
        super(Identity, self).__init__(
           linear = L.Linear(None, identityn, initialW=initializer)
        )

    def __call__(self, x):
        x = self.linear(x)
        return x


class D2AE(chainer.Chain):
    def __init__(self, config, encoder, identityn, device):
        super(D2AE, self).__init__()
        with self.init_scope():
            self.encoder = encoder
            self.decoder = Decoder(config)
            self.bt = Subnet()
            self.it = Identity(identityn)
            self.bp = Subnet()
            self.config=config
            self.device=device

    def statistical_augment(self,inputs, mean=0, stddev=0.01):
        with cupy.cuda.Device(self.device):        
            std=np.std(inputs.data,axis=0)
            std=chainer.cuda.to_cpu(std)
            std=np.tile(std, (len(inputs),1,1,1))
            noise=np.random.normal(loc=mean, scale=stddev, size=inputs.shape)
            noise=np.multiply(noise,std)
            #noise=chainer.cuda.to_gpu(Variable(noise))
            inputs_hat = inputs + xp.asarray(noise)
        return inputs_hat

    def __call__(self, x):
        x=self.encoder(x)
        ft=self.bt(x)
        yt=self.it(ft)
        fp=self.bp(x)
        ft_hat = self.statistical_augment(ft)
        fp_hat = self.statistical_augment(fp)
        cc=F.concat((ft,fp),axis=1)
        x_hat=F.concat((ft_hat,fp_hat),axis=1)
        x=self.decoder(cc)
        x_hat=self.decoder(x_hat)
        return x, x_hat, yt, fp