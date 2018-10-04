import chainer
from chainer import links as L
from chainer import functions as F
import numpy as np
import network

from chainer import training
from chainer.training import extension
from chainer.training import extensions
import updater.Updater


def make_adam_optimizer(model, alpha, beta1, beta2):
    optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1, beta2=beta2)
    optimizer.setup(model)
    return optimizer
def make_sgd_optimizer(model, lr):
    optimizer = cchainer.optimizers.MomentumSGD(lr=lr, momentum=0.9)
    optimizer.setup(model)
    return optimizer
def celebA_load():
    fn='share/datasets/CelebA/Eval/list_eval_partition.txt'    
    impath='share/datasets/CelebA/Img/img_align_celeba/'
    f = open(fn, "r");infos = f.readlines();f.close()
    fn='share/datasets/CelebA/Anno/identity_CelebA.txt'
    f = open(fn, "r");identity_infos = f.readlines();f.close()
    train_list=[]
    val_list=[]
    test_list=[]
    train_identity_label=[]
    val_identity_label=[]
    test_identity_label=[]
    for i in range(len(infos)):
        info=infos[i].split()
        identity_info=identity_infos[i].split()
        if int(info[1])==0:
            train_list.append(impath+info[0])
            train_identity_label.append(identity_info[1])
        if int(info[1])==1:
            val_list.append(impath+info[0])
            val_identity_label.append(identity_info[1])
        if int(info[1])==2:
            test_list.append(impath+info[0])
            test_identity_label.append(identity_info[1])
    return (train_list, val_list, test_list), (train_identity_list, val_identity_ist, test_identity_list)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--wt_dec", default=5e-4, type=float)
    parser.add_argument('--gpu', '-g', type=int, default=0, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result', help='Directory to output the result')
    args = parser.parse_args()

    train_dataset, _, _ = celebA_load()
    train_iter = chainer.iterators.SerialIterator(train_dataset, args.batchsize)

    opts = {}
    d2ae=network.D2AE().to_gpu()
    models = [d2ae]
    updater_args = {
        "iterator": {'main': train_iter},
        "device": args.gpu
    }

    opts["opt_d2ae"] = make_sgd_optimizer(d2ae, args.lr)    
    updater_args["optimizer"] = opts
    updater_args["models"] = models
    updater = Updater(**updater_args)

    trainer = training.Trainer(updater, (args.max_iter, 'iteration'), out=args.out)

    trainer.extend(extensions.LogReport(keys=report_keys,
                                        trigger=(args.display_interval, 'iteration')))
    trainer.extend(extensions.PrintReport(report_keys), trigger=(args.display_interval, 'iteration'))
    trainer.extend(extensions.ProgressBar(update_interval=10))

    # Run the training
    trainer.run()

