from __future__ import division

from chainer.training import extension


class MultistepShift(extension.Extension):

    """Trainer extension to shift an optimizer attribute in several steps.
    This extension changes an optimizer attribute in several steps, every step
    the attribute will multiply a factor ``gamma``.
    For example, suppose that this extension is called at every iteration,
    and ``init = x``, ``gamma = y``, ``step_value = [s1, s2, s3]``.
    Then during the iterations from 0 to (s1 - 1), the attr will be ``x``.
    During the iterations from s1 to (s2 - 1), the attr will be ``x * y``.
    During the iterations from s2 to (s3 - 1), the attr will be ``x * y * y``.
    During the iterations after s3, the attr will be ``x * y * y * y``.
    This extension is also called before the training loop starts by default.
    Args:
        attr (str): Name of the attribute to shift.
        init (float): Initial value of the attribute. If it is ``None``, the
            extension extracts the attribute at the first call and uses it as
            the initial value.
        gamma (float): The factor which the attr will mutiply at the beginning
            of each step.
        step_value (tuple): The first iterations of each step.
        optimizer (~chainer.Optimizer): Target optimizer to adjust the
            attribute. If it is ``None``, the main optimizer of the updater is
            used.
    """

    def __init__(self, attr, gamma, step_value, init, optimizer=None):
        self._attr = attr
        self._gamma = gamma
        self._step_value = step_value
        self._init = init
        self._optimizer = optimizer
        self._stepvalue_size = len(step_value)
        self._current_step = 0
        self._t = 0

    def initialize(self, trainer):
        optimizer = self._optimizer or trainer.updater.get_optimizer('main')
        if self._init is None:
            self._init = getattr(optimizer, self._attr)
        else:
            setattr(optimizer, self._attr, self._init)

    def __call__(self, trainer):
        self._t += 1
        optimizer = self._optimizer or trainer.updater.get_optimizer('main')
        if (self._current_step < self._stepvalue_size and
                self._t >= self._step_value[self._current_step]):
            self._current_step += 1
        value = self._init * pow(self._gamma, self._current_step)
        setattr(optimizer, self._attr, value)

    def serialize(self, serializer):
        self._t = serializer('_t', self._t)
        self._current_step = serializer('_current_step', self._current_step)

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

def copy_param(d2ae, encoder, identityn, config):
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
    tmp=Classification(encoder, identityn).to_gpu(config.gpu)
    serializers.load_npz(config.pretrainmodel_path, tmp)
    cnt=0
    cnt2=0
    for i,cpn in enumerate(tmp.namedparams()):
        cpn_sp=cpn[0].split('/')
        if cpn_sp[1]=='encoder':
            for j,tpn in enumerate(d2ae.namedparams()):
                if cpn[0]==tpn[0]:
                    #print(cpn[0],tpn[0])
                    tpn[1].data=cpn[1].data
                    cnt+=1
        #else:
        #    for j,tpn in enumerate(d2ae.namedparams()):
        #        if tpn[0]=='':  
        cnt2+=1
    print(cnt,cnt2)

def msceleb_loader(dir_path):
    indentityn=None 
    labels = os.listdir(dir_path)
    cnt=0
    training_data=[]
    for label in labels:
        #print(cnt, label)
        path = os.path.join(dir_path, label)
        images = os.listdir(path)
        for image_id in images:
            img_path = os.path.join(path, image_id)
            p=img_path.split('-')
            if int(p[2][0])==0:
                training_data.append((img_path, cnt))
        cnt+=1
    identityn=cnt
    return training_data, identityn

def celebA_rsa_loader(dir_path):
    impath='share/datasets/CelebA/rsa/data/0/'
    train_list=[]
    val_list=[]
    test_list=[]
    for i in range(72000):
            tmp='{:06d}'.format(i)
            train_list.append((impath+tmp+'.jpg',0))
            val_list.append((impath+tmp+'.jpg',0))
            test_list.append((impath+tmp+'.jpg',0))
    identityn=22512
    return train_list, val_list, test_list, identityn
