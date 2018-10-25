import chainer
from chainer import links as L
from chainer import functions as F
import numpy as np
#import network

from chainer import training
from chainer.training import extension
from chainer.training import extensions
from sklearn.model_selection import KFold
from sklearn.metrics.pairwise import cosine_similarity
from numpy import dot
from numpy.linalg import norm

def read_lfw_pairs(pairs_filename):
    pairs = []
    with open(pairs_filename, 'r') as f:
        for line in f.readlines()[1:]:
            pair = line.strip().split()
            pairs.append(pair)
    return np.array(pairs)

def get_lfw_paths(config, file_ext="jpg"):
    lfw_dir=config.lfw_dir
    pairs_path='pairs.txt'
    pairs = read_lfw_pairs(pairs_path)
    nrof_skipped_pairs = 0
    path_list0 = []
    path_list1 = []
    issame_list = []
    for i in range(len(pairs)):
        pair = pairs[i]
        if len(pair) == 3:
            path0 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])+'.'+file_ext)
            path1 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[2])+'.'+file_ext)
            issame = True
        elif len(pair) == 4:
            path0 = os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])+'.'+file_ext)
            path1 = os.path.join(lfw_dir, pair[2], pair[2] + '_' + '%04d' % int(pair[3])+'.'+file_ext)
            issame = False
        if os.path.exists(path0) and os.path.exists(path1):    # Only add the pair if both paths exist
            path_list0.append(path0)
            path_list1.append(path1)
            issame_list.append(issame)
        else:
            nrof_skipped_pairs += 1
    if nrof_skipped_pairs>0:
        print('Skipped %d image pairs' % nrof_skipped_pairs)
    return path_list0, path_list1, issame_list

def make_minibatches(data, minibatch_size = 2,  seed = 0, shuffle = 'random'):
    X0, X1, Y = data
    m = len(X0)
    minibatches = []
    num_complete_minibatches = math.floor(m/minibatch_size)
    for k in range(0, num_complete_minibatches):
        minibatch_X0 = X0[k * minibatch_size : k * minibatch_size + minibatch_size]
        minibatch_X1 = X1[k * minibatch_size : k * minibatch_size + minibatch_size]
        minibatch_Y = Y[k * minibatch_size : k * minibatch_size + minibatch_size]
        minibatches.append((minibatch_X0, minibatch_X1, minibatch_Y))

    rem_size = m - num_complete_minibatches * minibatch_size
    if m % minibatch_size != 0:
        minibatch_X0 = X0[num_complete_minibatches * minibatch_size : m]
        minibatch_X1 = X1[num_complete_minibatches * minibatch_size : m]
        minibatch_Y = Y[num_complete_minibatches * minibatch_size : m]
        minibatches.append((minibatch_X0, minibatch_X1, minibatch_Y))

    return minibatches

def batch2images(b):
    crop_size=235
    imgs=[]
    for i in range(len(b)):
        img=cv2.imread(b[i])
        img=img[:,:,::-1]
        img=cv2.resize(img,(crop_size,crop_size))
        img_in=img.transpose(2,0,1).astype(np.float32)
        img_in /= 255 
        img_in -= 0.5 # mean 0.5
        img_in *= 2.0 # std 0.5
        imgs.append(img_in)
    return imgs

class CosineDistance():
    def __init__(self, p):
        super(PairwiseDistance, self).__init__()
        self.norm = p

    def __call__(self, x1, x2,y):
        eps = 1e-4 / x1.shape[1]
        diff = np.abs(x1 - x2)
        out = np.sum(np.power(diff, self.norm),axis=1)
        out=np.power(out + eps, 1. / self.norm)
        distance=[]
        for i in range(x1.shape[0]):
            a=x1[i].flatten()
            b=x2[i].flatten()
            dis=1-dot(a, b)/(norm(a)*norm(b))
            distance.append(dis)
        distance=np.asarray(distance)
        distance=np.reshape(distance,(x1.shape[0],1,1))
        return distance

def evaluate(distances, labels, nrof_folds=10):
    # Calculate evaluation metrics
    thresholds = np.arange(0, 30, 0.01)
    tpr, fpr, accuracy = calculate_roc(thresholds, distances,
        labels, nrof_folds=nrof_folds)
    thresholds = np.arange(0, 30, 0.001)
    return tpr, fpr, accuracy

def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp+fn==0) else float(tp) / float(tp+fn)
    fpr = 0 if (fp+tn==0) else float(fp) / float(fp+tn)
    acc = float(tp+tn)/dist.size
    return tpr, fpr, acc

def calculate_roc(thresholds, distances, labels, nrof_folds=10):

    nrof_pairs = min(len(labels), len(distances))
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds,nrof_thresholds))
    fprs = np.zeros((nrof_folds,nrof_thresholds))
    accuracy = np.zeros((nrof_folds))

    indices = np.arange(nrof_pairs)

    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        # Find the best threshold for the fold
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(threshold, distances[train_set], labels[train_set])
        best_threshold_index = np.argmax(acc_train)
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx,threshold_idx], fprs[fold_idx,threshold_idx], _ = calculate_accuracy(threshold, distances[test_set], labels[test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(thresholds[best_threshold_index], distances[test_set], labels[test_set])

    tpr = np.mean(tprs,0)
    fpr = np.mean(fprs,0)
    return tpr, fpr, accuracy

if __name__ == '__main__':

    config=Config()
    args=Args()
    chainer.cuda.get_device(config.gpu)

    d2ae=D2AE(config, encoder).to_gpu(config.gpu)
    serializers.load_npz(config.pretrainmodel_path, d2ae)
    
    lfw_data = get_lfw_paths(config)
    minibatches = make_minibatches(lfw_data, minibatch_size = 2,  seed = 0, shuffle = 'sequential')

    l2_dist = CosineDistance(2)
    random.shuffle(minibatches)

    labels, distances, distancesp = [], [], []
    for i, cur_minibatch in enumerate(minibatches):
        print(i)
        x0, x1, y = cur_minibatch
        x0 = batch2images(x0)
        x1 = batch2images(x1)
        y=np.asarray(y, dtype=np.int32)
        y=np.reshape(y,(len(y),1))
        with cupy.cuda.Device(config.gpu):
            x0 = Variable(xp.asarray(x0, dtype=np.float32))
            x1 = Variable(xp.asarray(x1, dtype=np.float32))
        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            _,_,ft1,fp1 = d2ae(x1)
            _,_,ft0,fp0 = d2ae(x0)

        ft0=chainer.cuda.to_cpu(ft0.data)
        ft1=chainer.cuda.to_cpu(ft1.data)
        dists = l2_dist(ft0,ft1,y)#torch.sqrt(torch.sum((out_a - out_p) ** 2, 1))  # euclidean distance
        distances.append(dists)
        labels.append(y)

        fp0=chainer.cuda.to_cpu(fp0.data)
        fp1=chainer.cuda.to_cpu(fp1.data)
        distsp = l2_dist(fp0,fp1,y)#torch.sqrt(torch.sum((out_a - out_p) ** 2, 1))  # euclidean distance
        distancesp.append(distsp)
        
    labels = np.array([sublabel for label in labels for sublabel in label])
    distances = np.array([subdist[0] for dist in distances for subdist in dist])
    distancesp = np.array([subdist[0] for dist in distancesp for subdist in dist])

    tpr, fpr, accuracy = evaluate(distances,labels)
    print('ft',np.mean(accuracy))
    for i in range(len(tpr)):
        if fpr[i]>0.001:
            break
        print(i,tpr[i],fpr[i])
    print('tpr@0.001 ft', tpr[i])

    tpr, fpr, accuracy = evaluate(distancesp,labels)
    print('fp', np.mean(accuracy))
    for i in range(len(tpr)):
        if fpr[i]>0.001:
            break
        print(i,tpr[i],fpr[i])
    print('tpr@0.001 fp', tpr[i])
        





