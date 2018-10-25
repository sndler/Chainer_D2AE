
import numpy as np
import cupy
import chainer
import chainer.functions as F
from chainer import Variable


class Updater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.cl = kwargs.pop('models')
        self.config = kwargs.pop('config')
        super(Updater, self).__init__(*args, **kwargs)

    def get_optimizer_lr(self):
        if self.epoch<=10:
            return self.config.lr
        elif self.epoch<=20:
            return self.config.lr*0.1
        elif self.epoch<=30:
            return self.config.lr*0.1*0.1
        else:
            return self.config.lr*0.1*0.1*0.1

    def update_core(self):
        optimizer = self.get_optimizer('opt')
        xp = self.cl.xp

        batch, batch_out, label = self.get_iterator('main').next()
        batchsize = len(batch)
        x = []; 
        for i in range(batchsize):
            x.append(np.asarray(batch[i]).astype("f"))
        with cupy.cuda.Device(self.config.gpu):
            x_in = Variable(xp.asarray(x))
            label = Variable(xp.asarray(label, dtype=np.int32))

        out = self.cl(x_in)
        loss = F.softmax_cross_entropy(out,label)
        self.cl.cleargrads()
        loss.backward()
        optimizer.update()
        chainer.reporter.report({'loss': loss})
        optimizer.lr = self.get_optimizer_lr()


