
import numpy as np

import chainer
import chainer.functions as F
from chainer import Variable


class Updater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.d2ae, self.ip = kwargs.pop('models')
        super(Updater, self).__init__(*args, **kwargs)

    def update_core(self):
        d2ae_optimizer = self.get_optimizer('opt_d2ae')
        ip_optimizer = self.get_optimizer('opt_ip')
        xp = self.d2ae.xp

        batch, batch_out, label = self.get_iterator('main').next()
        batchsize = len(batch)
        x = []; x_out=[]
        for i in range(batchsize):
            x.append(np.asarray(batch[i]).astype("f"))
            x_out.append(np.asarray(batch_out[i]).astype("f"))
        x_in = Variable(xp.asarray(x))
        x_out = Variable(xp.asarray(x_out))
        label = Variable(xp.asarray(label, dtype=np.int32))

        X,X_hat,yt,fp = self.d2ae(x_in)
        fp=Variable(fp.data)
        yp=self.ip(fp)
        loss_I = F.softmax_cross_entropy(yt,label)
        loss_X = F.mean_squared_error(X,x_out)
        loss_Xh = F.mean_squared_error(X_hat,x_out)
        loss_adv = F.softmax_cross_entropy(yp,label)
        loss_H = F.softmax_cross_entropy(-yp,label)
        loss = loss_I + (loss_adv + loss_H) + (loss_X + loss_Xh)
        
        self.d2ae.cleargrads()
        self.ip.cleargrads()
        loss.backward()
        d2ae_optimizer.update()
        ip_optimizer.update()
