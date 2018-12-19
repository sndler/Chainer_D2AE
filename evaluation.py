import chainer
import random
import numpy as np
import cupy
import chainer.functions as F
from chainer import Variable
import cv2





def eval_classification(dataset, d2ae, ip, evaln, batchn, device,msg='tmp'):
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
        xp = d2ae.xp
        mean_top10_acc1=0
        mean_top100_acc1=0
        mean_top1000_acc1=0
        mean_top10_acc2=0
        mean_top100_acc2=0
        mean_top1000_acc2=0
        for i in range(evaln):
            start=random.randint(0,100)
            start=min(len(dataset)-batchn,start)
            end=start+batchn
            batch, label = load_image(dataset,start,end)
            batchsize = len(batch)
            x = []; 
            for i in range(batchsize):
                x.append(np.asarray(batch[i]).astype("f"))
            with cupy.cuda.Device(device):
                x_in = Variable(xp.asarray(x))
            with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
                X,X_hat,yt,fp = d2ae(x_in)
                yp=ip(fp)
                yt=F.softmax(yt)
                yp=F.softmax(yp)
            label = np.asarray(label, dtype=np.int32)
            label = np.reshape(label, (label.shape[0],1))
            #top1_acc = get_acc(chainer.cuda.to_cpu(out.data),label,topk=1)
            top10_acc1 = get_acc(chainer.cuda.to_cpu(yt.data),label,topk=10)
            top100_acc1 = get_acc(chainer.cuda.to_cpu(yt.data),label,topk=100)
            top1000_acc1 = get_acc(chainer.cuda.to_cpu(yt.data),label,topk=1000)
            #mean_top1_acc += top1_acc/batchn
            mean_top10_acc1 += top10_acc1
            mean_top100_acc1 += top100_acc1
            mean_top1000_acc1 += top1000_acc1

            top10_acc2 = get_acc(chainer.cuda.to_cpu(yp.data),label,topk=10)
            top100_acc2 = get_acc(chainer.cuda.to_cpu(yp.data),label,topk=100)
            top1000_acc2 = get_acc(chainer.cuda.to_cpu(yp.data),label,topk=1000)
            #mean_top1_acc += top1_acc/batchn
            mean_top10_acc2 += top10_acc2
            mean_top100_acc2 += top100_acc2
            mean_top1000_acc2 += top1000_acc2

        mean_top10_acc1 /= evaln
        mean_top100_acc1 /= evaln
        mean_top1000_acc1 /= evaln
        mean_top10_acc2 /= evaln
        mean_top100_acc2 /= evaln
        mean_top1000_acc2 /= evaln
        #print('top1',mean_top1_acc)
        print('yt',msg)
        print('top10score: ',mean_top10_acc1) 
        print('top100score: ',mean_top100_acc1) 
        print('top1000score: ',mean_top1000_acc1)
        print('yp',msg)
        print('top10score: ',mean_top10_acc2) 
        print('top100score: ',mean_top100_acc2) 
        print('top1000score: ',mean_top1000_acc2)
    return evaluation

def eval_reconstruction(dataset, d2ae, batchn, device,msg='tmp', out='results'):
    def load_image(dataset, i,i_end, random_crop=False):
        imgs=[]; imgs_out=[]; labels=[]; imgs_org=[];
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
            #print(img.shape)
            imgs_org.append(img[:,:,::-1])
            labels.append(label)
        return (np.asarray(imgs), labels, np.asarray(imgs_org))

    @chainer.training.make_extension()
    def evaluation(trainer):
        xp = d2ae.xp
        for i in range(1):
            start=random.randint(0,100)
            start=min(len(dataset)-batchn,start)
            end=start+batchn
            batch, label,img_org = load_image(dataset,start,end)
            batchsize = len(batch)
            x = []; 
            for i in range(batchsize):
                x.append(np.asarray(batch[i]).astype("f"))
            with cupy.cuda.Device(device):
                x_in = Variable(xp.asarray(x))
            with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
                X,X_hat,yt,fp = d2ae(x_in)
            X=chainer.cuda.to_cpu(X.data)
            X=X.transpose(0,2,3,1)
            X = 0.5 * (X + 1)  # [-1,1] => [0, 1]
            X = np.clip(X,0, 1)
            #print(X.shape)
            X=X[0,:,:,::-1]
            X=(X*255).astype(np.uint8)
            cv2.imwrite(out+'/tmp'+str(trainer.updater.iteration)+'.png',X)

            X=chainer.cuda.to_cpu(x_in.data)
            X=X.transpose(0,2,3,1)
            X = 0.5 * (X + 1)  # [-1,1] => [0, 1]
            X = np.clip(X,0, 1)
            #print(X.shape)
            X=X[0,:,:,::-1]
            X=(X*255).astype(np.uint8)
            cv2.imwrite(out+'/org'+str(trainer.updater.iteration)+'.png',X)
    return evaluation
