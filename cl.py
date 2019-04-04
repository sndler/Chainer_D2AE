import chainer
from chainer import links as L
from chainer import functions as F
import numpy as np
import argparse
from chainer import serializers

import network

from chainer import training
from chainer.training import extension
from chainer.training import extensions
from updater_cl import Updater
from iterator import MyIterator
from encoder import InceptionResNetV2
from util import MultistepShift

#from config import Config
class Config(object):
    celebA_dir = '/root/share/datasets/CelebA/'
    ms_celeb_dir='/root/share/datasets/MSCeleb1M/raw'
    #pretrainmodel_path='inception_resnet.npz'
    #pretrainmodel_path='result/Classification_40000.npz'
    pretrainmodel_path='/root/inception_resnet.npz'
    display_interval=10
    lr=0.01
    gpu=6
    batchsize=32*3
    max_iter= 1184160/batchsize*30
    epoch_iter= 1184160/batchsize
    #max_iter= 162770/batchsize*30
    evaluation_interval=100
    out='msceleb_pretrain3'
    display_interval=10
    snapshot_interval=50000

def eval_classification(dataset, cl, evaln, batchn, gpu,msg='tmp'):
    def load_image(dataset, i,i_end, random_crop=False):
        imgs=[]; imgs_out=[]; labels=[]
        crop_size=235
        for i in range(i,i_end):
            imn, label=dataset[i]
            img=cv2.imread(imn)
            img=img[:,:,::-1]
            img_in=cv2.resize(img,(crop_size,crop_size)).astype(np.float32)
            img_in /= 255.
            img_in -= 0.5 # mean 0.5
            img_in *= 2.0 # std 0.5
            img_in=img_in.transpose(2,0,1).astype(np.float32)
            imgs.append(img_in)
            labels.append(label)
        return (np.asarray(imgs), labels)

    def get_acc(out,label,topk=1):
        sortid=np.argsort(-out)
        correctn=np.sum(sortid[:,0:topk]==label)
        topk_acc=float(correctn)/(len(out))
        return topk_acc

    @chainer.training.make_extension()
    def evaluation(trainer):
        xp = cl.xp
        mean_top10_acc1=0
        mean_top100_acc1=0
        mean_top1000_acc1=0
        for i in range(evaln):
            start=random.randint(0,100)
            start=min(len(dataset)-batchn,start)
            end=start+batchn
            batch, label = load_image(dataset,start,end)
            batchsize = len(batch)
            x = []; 
            for i in range(batchsize):
                x.append(np.asarray(batch[i]).astype("f"))
            with cupy.cuda.Device(gpu):
                x_in = Variable(xp.asarray(x))
            with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
                y = cl(x_in)
                y=F.softmax(y)
            label = np.asarray(label, dtype=np.int32)
            label = np.reshape(label, (label.shape[0],1))
            #top1_acc = get_acc(chainer.cuda.to_cpu(out.data),label,topk=1)
            top10_acc1 = get_acc(chainer.cuda.to_cpu(y.data),label,topk=10)
            top100_acc1 = get_acc(chainer.cuda.to_cpu(y.data),label,topk=100)
            top1000_acc1 = get_acc(chainer.cuda.to_cpu(y.data),label,topk=1000)
            #mean_top1_acc += top1_acc/batchn
            mean_top10_acc1 += top10_acc1
            mean_top100_acc1 += top100_acc1
            mean_top1000_acc1 += top1000_acc1

        mean_top10_acc1 /= evaln
        mean_top100_acc1 /= evaln
        mean_top1000_acc1 /= evaln
        print('y',msg)
        print('top10score: ',mean_top10_acc1) 
        print('top100score: ',mean_top100_acc1) 
        print('top1000score: ',mean_top1000_acc1)
    return evaluation


class Classification(chainer.Chain):
    def __init__(self, encoder, identityn):
        super(Classification, self).__init__()
        with self.init_scope():
            self.encoder = InceptionResNetV2()
            self.last_linear = L.Linear(None, identityn)
    def __call__(self, x):
        x=self.encoder(x)
        x = F.average_pooling_2d(x, (x.shape[2],x.shape[3]))
        x=self.last_linear(x)
        return x

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--wt_dec", default=5e-4, type=float)
    parser.add_argument('--gpu', '-g', type=int, default=0, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--lr', '-l', type=int, default=1e-3, help='learning rate')
    parser.add_argument('--out', '-o', default='result', help='Directory to output the result')
    args = parser.parse_args()
    config=Config()

    train_dataset, _, _, identityn = celebA_load(config)
    train_iter = MyIterator(config, train_dataset, args.batch_size)

    opts = {}
    encoder=InceptionResNetV2()
    serializers.load_npz(config.pretrainmodel_path, encoder)
    clmodel=Classification(encoder, identityn).to_gpu(config.gpu)
    models = clmodel
    updater_args = {
        "iterator": {'main': train_iter},
        "device": args.gpu
    }

    opts["opt"] = make_sgd_optimizer(clmodel, config.lr)    
    updater_args["optimizer"] = opts
    updater_args["models"] = models
    updater_args["config"] = config
    updater = Updater(**updater_args)
    report_keys = ["loss"]

    trainer = training.Trainer(updater, (config.max_iter, 'iteration'), out=config.out)
    trainer.extend(MultistepShift('lr', 0.1, [config.epoch_iter*10,config.epoch_iter*20], 1e-2, optimizer=opts["opt"]))
    trainer.extend(extensions.snapshot_object(
                clmodel, clmodel.__class__.__name__ + '_{.updater.iteration}.npz'), trigger=(config.snapshot_interval, 'iteration'))
    trainer.extend(extensions.LogReport(keys=report_keys,
                                            trigger=(config.display_interval, 'iteration')))
    trainer.extend(extensions.PrintReport(report_keys), trigger=(config.display_interval, 'iteration'))
    trainer.extend(extensions.ProgressBar(update_interval=10))
    trainer.extend(eval_classification(train_dataset, clmodel, 1, 100, config.gpu,msg='train'), trigger=(config.evaluation_interval, 'iteration'),
                priority=extension.PRIORITY_WRITER)

    # Run the training
    trainer.run()

