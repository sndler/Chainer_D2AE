
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


