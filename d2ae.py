import chainer
from chainer import links as L
from chainer import functions as F
import numpy as np
import network

from chainer import training
from chainer.training import extension
from chainer.training import extensions
from updater_cl import Updater
from iterator import MyIterator
from encoder import InceptionResNetV2
from config import Config
from util import MultistepShift


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
    trainer.extend(MultistepShift('lr', 0.1, [args.epoch_iter*10,args.epoch_iter*20], 1e-2, optimizer=opts["opt_d2ae"]))
    trainer.extend(MultistepShift('lr', 0.1, [args.epoch_iter*10,args.epoch_iter*20], 1e-2, optimizer=opts["opt_ip"]))
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

