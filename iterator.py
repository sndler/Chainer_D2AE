from __future__ import division

import numpy
import cv2
from chainer.dataset import iterator
import random

class MyIterator(iterator.Iterator):

    def __init__(self, config, dataset, batch_size, repeat=True, shuffle=True):
        self.config = config
        self.dataset = dataset
        self.batch_size = batch_size
        self._repeat = repeat
        self._shuffle = shuffle
        self.crop_size=235
        self.resize_shape=(int(235*1.05),int(235*1.05))
        self.random_crop=True

        self.reset()
    def load_image(self, i,i_end):
        imgs=[]; imgs_out=[]; labels=[]
        for i in range(i,i_end):
            if i>=(len(self._order)-1):
                break
            imn, label=self.dataset[self._order[i]]
            img=cv2.imread(imn)
            img=img[:,:,::-1]
            #img=cv2.resize(img,self.resize_shape)
            img=cv2.resize(img,(self.crop_size,self.crop_size))
            if self.random_crop==True:
                h,w=self.resize_shape
                top = random.randint(0, h - self.crop_size - 1)
                left = random.randint(0, w - self.crop_size - 1)
                bottom = top + self.crop_size
                right = left + self.crop_size
                #img = img[top:bottom, left:right, :]
                if random.random()>0.5:
                    img=img[:,::-1,:]
            img_in=img.transpose(2,0,1).astype(np.float32)
            img_in /= 255 
            img_in -= 0.5 # mean 0.5
            img_in *= 2.0 # std 0.5
            imgs.append(img_in)

            img_out=cv2.resize(img,(320,320)).transpose(2,0,1)
            img_out = img_out.astype(np.float32)/255
            img_out -= 0.5
            img_out *= 2.0
            imgs_out.append(img_out)
            labels.append(label)
        return (np.asarray(imgs), np.asarray(imgs_out), labels)

    def __next__(self):
        if not self._repeat and self.epoch > 0:
            raise StopIteration

        self._previous_epoch_detail = self.epoch_detail

        i = self.current_position
        i_end = i + self.batch_size
        N = len(self.dataset)

        if self._order is None:
            batch = self.load_image(i,i_end)
        else:
            batch = self.load_image(i,i_end)

        if i_end >= N:
            if self._repeat:
                rest = i_end - N
                if self._order is not None:
                    numpy.random.shuffle(self._order)
                if rest > 0:
                    batch = self.load_image(0,self.batch_size)
                self.current_position = self.batch_size
            else:
                self.current_position = 0

            self.epoch += 1
            self.is_new_epoch = True
        else:
            self.is_new_epoch = False
            self.current_position = i_end

        return batch

    next = __next__

    @property
    def epoch_detail(self):
        return self.epoch + self.current_position / len(self.dataset)

    @property
    def previous_epoch_detail(self):
        if self._previous_epoch_detail < 0:
            return None
        return self._previous_epoch_detail

    def serialize(self, serializer):
        self.current_position = serializer('current_position',
                                           self.current_position)
        self.epoch = serializer('epoch', self.epoch)
        self.is_new_epoch = serializer('is_new_epoch', self.is_new_epoch)
        if self._order is not None:
            try:
                serializer('order', self._order)
            except KeyError:
                serializer('_order', self._order)
        try:
            self._previous_epoch_detail = serializer(
                'previous_epoch_detail', self._previous_epoch_detail)
        except KeyError:
            # guess previous_epoch_detail for older version
            self._previous_epoch_detail = self.epoch + \
                (self.current_position - self.batch_size) / len(self.dataset)
            if self.epoch_detail > 0:
                self._previous_epoch_detail = max(
                    self._previous_epoch_detail, 0.)
            else:
                self._previous_epoch_detail = -1.

    def reset(self):
        if self._shuffle:
            self._order = numpy.random.permutation(len(self.dataset))
        else:
            self._order = None

        self.current_position = 0
        self.epoch = 0
        self.is_new_epoch = False

        # use -1 instead of None internally.
        self._previous_epoch_detail = -1.

    @property
    def repeat(self):
        return self._repeat
