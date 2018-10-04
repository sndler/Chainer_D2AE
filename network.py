
import chainer
from chainer import links as L
from chainer import functions as F
import numpy as np
import cupy as xp
import encoder.Encoder
import decoder.Decoder

from chainer import Variable
import chainer.cuda

class Subnet(chainer.Chain):
    def __init__(self):
        super(Subnet, self).__init__(
            conv1 = L.Convolution2D(2048, 256, 3,1,pad=1),
            bn1 = L.BatchNormalization(256),
            conv2 = L.Convolution2D(256, 256, 3,1,pad=1),
            bn2 = L.BatchNormalization(256),
            conv3 = L.Convolution2D(256, 256, 3,1,pad=1),
            bn3 = L.BatchNormalization(256),
            fc = L.Linear(None, 256)
            bn4 = L.BatchNormalization(256),
        )

    def __call__(self, x):
        print(x.shape,type(x))
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.average_pooling_2d(x, (x.shape[2],x.shape[3]))
        x = F.relu(self.bn3(self.fc(x)))
        #x = F.reshape(x, (x.shape[0],x.shape[1]))
        return x

class Identity(chainer.Chain):
    def __init__(self, identityn):
        super(Identity, self).__init__(
           linear = L.Linear(None, identityn)
        )

    def __call__(self, x):
        x = self.linear(x)
        return x


class D2AE(chainer.Chain):
    def __init__(self, identityn):
        super(D2AE, self).__init__()
        with self.init_scope():
            self.encoder = Encoder()
            self.decoder = Decoder()
            self.bt = Subnet()
            self.it = Identity(identityn)
            self.bp = Subnet()

    def statistical_augment(self,inputs, mean=0, stddev=0.01):
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
        print(ft.shape)
        yt=self.it(ft)
        fp=self.bp(x)
        ft_hat = self.statistical_augment(ft)
        fp_hat = self.statistical_augment(fp)
        x=F.concat((ft,fp),axis=1)
        x_hat=F.concat((ft_hat,fp_hat),axis=1)
        x=self.decoder(x)
        x_hat=self.decoder(x_hat)
        return x, x_hat, yt, fp