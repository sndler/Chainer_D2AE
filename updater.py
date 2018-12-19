
import numpy as np

import chainer
import chainer.functions as F
from chainer import Variable
import cupy
from chainer.dataset import convert

class Updater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.d2ae, self.ip = kwargs.pop('models')
        self.config = kwargs.pop('config')
        self.lamdat=1
        self.lamdap=0.1
        self.lamdax= 1.81e-5
        self.convert = convert.concat_examples
        
        super(Updater, self).__init__(*args, **kwargs)

    def update_core(self):
        d2ae_optimizer = self.get_optimizer('opt_d2ae')
        ip_optimizer = self.get_optimizer('opt_ip')
        xp = self.d2ae.xp

        x_in, x_out, label = self.get_iterator('main').next()
        x_in = self.converter(x_in, self.device)
        x_out = self.converter(x_out, self.device)
        label = self.converter(label, self.device)
        x_in=xp.array(x_in)
        x_out=xp.array(x_out)
        label=xp.array(label)
        

        X,X_hat,yt,fp = self.d2ae(x_in)
        yp=self.ip(fp)
        loss_I = F.softmax_cross_entropy(yt,label)
        loss_X = F.mean_squared_error(X,x_out)*X.shape[1]*X.shape[2]*X.shape[3]
        loss_Xh = F.mean_squared_error(X_hat,x_out)*X.shape[1]*X.shape[2]*X.shape[3]
        loss_H = F.softmax_cross_entropy(-yp,label)
        loss1 = self.lamdat*loss_I + self.lamdap*(loss_H) + self.lamdax*(loss_X + loss_Xh)
        self.d2ae.cleargrads()
        loss1.backward()
        d2ae_optimizer.update()
        chainer.reporter.report({'loss_I': loss_I})        
        chainer.reporter.report({'loss_X': self.lamdax*loss_X})        
        chainer.reporter.report({'loss1': loss1})

        self.ip.cleargrads()
        loss_adv = F.softmax_cross_entropy(yp,label)
        loss2 = self.lamdap*(loss_adv)
        loss2.backward()
        ip_optimizer.update()
        chainer.reporter.report({'loss_adv': self.lamdap*loss_adv})        
        chainer.reporter.report({'loss_H': self.lamdap*loss_H})
        chainer.reporter.report({'loss2': loss2})        
