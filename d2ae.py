import chainer
from chainer import links as L
from chainer import functions as F
import numpy as np
import network

from chainer import training
from chainer.training import extension
from chainer.training import extensions
import updater.Updater
import iterator.MyIterator
from config import Config


def make_adam_optimizer(model, alpha, beta1, beta2):
    optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1, beta2=beta2)
    optimizer.setup(model)
    return optimizer
def make_sgd_optimizer(model, lr):
    optimizer = chainer.optimizers.MomentumSGD(lr=lr, momentum=0.9)
    optimizer.setup(model)
    return optimizer
def celebA_load(config):
    fn=config.celevA_dir+'Eval/list_eval_partition.txt'    
    impath=config.celevA_dir+'Img/img_align_celeba/'
    f = open(fn, "r");infos = f.readlines();f.close()
    fn=config.celevA_dir+'Anno/identity_CelebA.txt'
    f = open(fn, "r");identity_infos = f.readlines();f.close()
    train_list=[]
    val_list=[]
    test_list=[]
    all_labels=[]
    for i in range(len(infos)):
        info=infos[i].split()
        identity_info=identity_infos[i].split()
        if int(info[1])==0:
            train_list.append((impath+info[0],identity_info[1]))
        if int(info[1])==1:
            val_list.append((impath+info[0],identity_info[1]))
        if int(info[1])==2:
            test_list.append((impath+info[0],identity_info[1]))
        all_labels.append(int(identity_info[1]))
    identityn=max(all_labels)
    return train_list, val_list, test_list, identityn

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--wt_dec", default=5e-4, type=float)
    parser.add_argument('--gpu', '-g', type=int, default=0, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result', help='Directory to output the result')
    args = parser.parse_args()

    config=Config()

    train_dataset, _, _, identityn = celebA_load(config)
    train_iter = MyIterator(config, train_dataset, args.batchsize)

    opts = {}
    encoder=InceptionResNetV2()
    serializers.load_npz(config.pretrainmodel_path, encoder)
    d2ae=network.D2AE(encoder, identityn).to_gpu()
    ip=network.Identity(identityn).to_gpu()

    models = [d2ae,ip]
    updater_args = {
        "iterator": {'main': train_iter},
        "device": args.gpu
    }

    opts["opt_d2ae"] = make_sgd_optimizer(d2ae, args.lr)
    opts["opt_ip"] = make_sgd_optimizer(ip, args.lr)
    updater_args["optimizer"] = opts
    updater_args["models"] = models
    updater_args["config"] = config
    updater = Updater(**updater_args)
    report_keys = ["loss1","loss2","loss_I","loss_X","loss_adv", "loss_H"]

    trainer = training.Trainer(updater, (args.max_iter, 'iteration'), out=args.out)

    trainer.extend(extensions.LogReport(keys=report_keys,
                                        trigger=(args.display_interval, 'iteration')))
    trainer.extend(extensions.snapshot_object(
                d2ae, d2ae.__class__.__name__ + '_{.updater.iteration}.npz'), trigger=(args.snapshot_interval, 'iteration'))
    trainer.extend(extensions.PrintReport(report_keys), trigger=(args.display_interval, 'iteration'))
    trainer.extend(extensions.ProgressBar(update_interval=10))
    trainer.extend(eval_classification(train_dataset, d2ae, ip, 1, 10, config.gpu,msg='train'), trigger=(args.evaluation_interval, 'iteration'),
               priority=extension.PRIORITY_WRITER)
    trainer.extend(eval_reconstruction(train_dataset, d2ae, 10, config.gpu,msg='train', out=args.out), trigger=(args.evaluation_interval, 'iteration'),
               priority=extension.PRIORITY_WRITER)

    # Run the training
    trainer.run()

