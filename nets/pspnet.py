import pickle
import numpy as np
from PIL import Image
import cv2
import os
import chainer
from chainer import cuda
from chainer import optimizers
from chainer import serializers
from chainer import Variable
import chainer.functions as F
import chainer.links as L
from progressbar import ProgressBar
from base_network import BaseNetwork

class BottleNeck_proj(chainer.Chain):
    def __init__(self, in_channel, ch, out_channel):
        super(BottleNeck_proj, self).__init__(
            conv1=L.Convolution2D(in_channel, ch, 1, stride=1, pad=0, nobias=True),
            bn1=L.BatchNormalization(ch),
            conv2=L.Convolution2D(ch, ch, 3, stride=1, pad=1, nobias=True),
            bn2=L.BatchNormalization(ch),
            conv3=L.Convolution2D(ch, out_channel, 1, stride=1, pad=0, nobias=True),
            bn3=L.BatchNormalization(out_channel),
        )

    def __call__(self, x, y, test=False):
        h = F.relu(self.bn1(self.conv1(x), test=test))
        h = F.relu(self.bn2(self.conv2(h), test=test))
        h = self.bn3(self.conv3(h), test=test)

        return F.relu(h + y)


class BottleNeck(chainer.Chain):
    def __init__(self, in_channel, ch, out_channel):
        super(BottleNeck, self).__init__(
            conv1=L.Convolution2D(in_channel, ch, 1, stride=1, pad=0, nobias=True),
            bn1=L.BatchNormalization(ch),
            conv2=L.Convolution2D(ch, ch, 3, stride=1, pad=1, nobias=True),
            bn2=L.BatchNormalization(ch),
            conv3=L.Convolution2D(ch, out_channel, 1, stride=1, pad=0, nobias=True),
            bn3=L.BatchNormalization(out_channel),
        )

    def __call__(self, x, test=False):
        h = F.relu(self.bn1(self.conv1(x), test=test))
        h = F.relu(self.bn2(self.conv2(h), test=test))
        h = self.bn3(self.conv3(h), test=test)

        return F.relu(h + x)

class BottleNeck_dilated(chainer.Chain):
    def __init__(self, in_channel, ch, out_channel, dilate):
        super(BottleNeck_dilated, self).__init__(
            conv1=L.Convolution2D(in_channel, ch, 1, stride=1, pad=0, nobias=True),
            bn1=L.BatchNormalization(ch),
            conv2=L.DilatedConvolution2D(ch, ch, 3, dilate=dilate,  stride=1, pad=dilate, nobias=True),
            bn2=L.BatchNormalization(ch),
            conv3=L.Convolution2D(ch, out_channel, 1, stride=1, pad=0, nobias=True),
            bn3=L.BatchNormalization(out_channel),
        )

    def __call__(self, x, test=False):
        h = F.relu(self.bn1(self.conv1(x), test=test))
        h = F.relu(self.bn2(self.conv2(h), test=test))
        h = self.bn3(self.conv3(h), test=test)

        return F.relu(h + x)

class Block(chainer.Chain):
    def __init__(self, layer, in_size, ch, out_size, Dilate=1):
        super(Block, self).__init__()
        links = []
        if Dilate!=1:
            for i in range(layer):
                links += [('res{}'.format(i+1), BottleNeck_dilated(in_size, ch, out_size, Dilate))]
        else:
            for i in range(layer):
                links += [('res{}'.format(i+1), BottleNeck(in_size, ch, out_size))]

        for l in links:
            self.add_link(*l)
        self.forward = links

    def __call__(self, x, test=False):
        for name, _ in self.forward:
            f = getattr(self, name)
            x = f(x, test)

        return x


class PSPNet(chainer.Chain):
    def __init__(self, in_channel, out_channel, input_height, input_width, gpu1, gpu2, gpu3):
        self.input_height = input_height
        self.input_width = input_width
        self.gpu1 = gpu1
        self.gpu2 = gpu2
        self.gpu3 = gpu3
        super(PSPNet, self).__init__(
            conv1_1=L.Convolution2D(in_channel, 64, 3, stride=2, pad=1, nobias=True).to_gpu(gpu1),
            bn1_1=L.BatchNormalization(64).to_gpu(gpu1),
            conv1_2=L.Convolution2D(64, 64, 3, stride=1, pad=1, nobias=True).to_gpu(gpu1),
            bn1_2=L.BatchNormalization(64).to_gpu(gpu1),
            conv1_3=L.Convolution2D(64, 128, 3, stride=1, pad=1, nobias=True).to_gpu(gpu1),
            bn1_3=L.BatchNormalization(128).to_gpu(gpu1),

            conv2_1_proj = L.Convolution2D(128, 256, 1, stride=1, pad=0, nobias=True).to_gpu(gpu1),
            bn2_1_proj=L.BatchNormalization(256).to_gpu(gpu1),
            conv2_1_res = BottleNeck_proj(128, 64, 256).to_gpu(gpu1),
            conv2_res = Block(2, 256, 64, 256).to_gpu(gpu1),

            conv3_1_proj = L.Convolution2D(256, 512, 1, stride=1, pad=0, nobias=True).to_gpu(gpu1),
            bn3_1_proj=L.BatchNormalization(512).to_gpu(gpu1),
            conv3_1_res = BottleNeck_proj(256, 128, 512).to_gpu(gpu1),
            conv3_res = Block(3, 512, 128, 512).to_gpu(gpu2),

            conv4_1_proj = L.Convolution2D(512, 1024, 1, stride=1, pad=0, nobias=True).to_gpu(gpu1),
            bn4_1_proj=L.BatchNormalization(1024).to_gpu(gpu1),
            conv4_1_res = BottleNeck_proj(512, 256, 1024).to_gpu(gpu1),
            conv4_res = Block(7, 1024, 256, 1024, Dilate=2).to_gpu(gpu2),
            conv4_res2 = Block(15, 1024, 256, 1024, Dilate=2).to_gpu(gpu3),


            conv5_1_proj = L.Convolution2D(1024, 2048, 1, stride=1, pad=0, nobias=True).to_gpu(gpu1),
            bn5_1_proj=L.BatchNormalization(2048).to_gpu(gpu1),
            conv5_1_res = BottleNeck_proj(1024, 512, 2048).to_gpu(gpu1),
            conv5_res = Block(2, 2048, 512, 2048, Dilate=4).to_gpu(gpu2),

            conv5_3_pool1_conv = L.Convolution2D(2048, 512, 1, stride=1, pad=0, nobias=True).to_gpu(gpu1),
            bn5_3_pool1_conv=L.BatchNormalization(512).to_gpu(gpu1),
            conv5_3_pool2_conv = L.Convolution2D(2048, 512, 1, stride=1, pad=0, nobias=True).to_gpu(gpu1),
            bn5_3_pool2_conv=L.BatchNormalization(512).to_gpu(gpu1),
            conv5_3_pool3_conv = L.Convolution2D(2048, 512, 1, stride=1, pad=0, nobias=True).to_gpu(gpu1),
            bn5_3_pool3_conv=L.BatchNormalization(512).to_gpu(gpu1),
            conv5_3_pool6_conv = L.Convolution2D(2048, 512, 1, stride=1, pad=0, nobias=True).to_gpu(gpu1),
            bn5_3_pool6_conv=L.BatchNormalization(512).to_gpu(gpu1),

            conv5_4 = L.Convolution2D(4096, 512, 3, stride=1, pad=1, nobias=True).to_gpu(gpu1),
            bn5_4=L.BatchNormalization(512).to_gpu(gpu1),

            conv6 = L.Convolution2D(512, out_channel, 1, stride=1, pad=0, nobias=True).to_gpu(gpu1),


        )


    def __call__(self, x, train=True, test=False):
        conv1_1 = F.relu(self.bn1_1(self.conv1_1(x), test=test))
        conv1_2 = F.relu(self.bn1_2(self.conv1_2(conv1_1), test=test))
        conv1_3 = F.relu(self.bn1_3(self.conv1_3(conv1_2), test=test))
        pool1 = F.max_pooling_2d(conv1_3,3,stride=2)

        conv2_1_proj = self.bn2_1_proj(self.conv2_1_proj(pool1), test=test)
        conv2_1 = self.conv2_1_res(pool1, conv2_1_proj, test=test)
        conv2_3 = self.conv2_res(conv2_1, test=test)

        conv3_1_proj = self.bn3_1_proj(self.conv3_1_proj(F.copy(conv2_3,self.gpu1)), test=test)
        conv3_1 = self.conv3_1_res(F.copy(conv2_3,self.gpu1), conv3_1_proj, test=test)
        conv3_4 = self.conv3_res(F.copy(conv3_1,self.gpu2), test=test)

        conv4_1_proj = self.bn4_1_proj(self.conv4_1_proj(F.copy(conv3_4,self.gpu1)), test=test)
        conv4_1 = self.conv4_1_res(F.copy(conv3_4,self.gpu1), conv4_1_proj, test=test)
        conv4_23_1 = self.conv4_res(F.copy(conv4_1, self.gpu2), test=test)
        conv4_23 = self.conv4_res2(F.copy(conv4_23_1, self.gpu3), test=test)

        conv5_1_proj = self.bn5_1_proj(self.conv5_1_proj(F.copy(conv4_23,self.gpu1)), test=test)
        conv5_1 = self.conv5_1_res(F.copy(conv4_23,self.gpu1), conv5_1_proj, test=test)
        conv5_3 = self.conv5_res(F.copy(conv5_1,self.gpu2), test=test)
        coonv5_3 = F.copy(conv5_3,self.gpu1)

        _,c,h,w = conv5_3.data.shape
        conv5_3_pool1 = F.average_pooling_2d(coonv5_3, h, stride=h)
        conv5_3_pool1_conv = self.bn5_3_pool1_conv(self.conv5_3_pool1_conv(conv5_3_pool1), test=test)
        conv5_3_pool1_interp = F.resize_images(conv5_3_pool1_conv, (h,w))

        conv5_3_pool2 = F.average_pooling_2d(coonv5_3, h/2, stride=h/2)
        conv5_3_pool2_conv = self.bn5_3_pool2_conv(self.conv5_3_pool2_conv(conv5_3_pool2), test=test)
        conv5_3_pool2_interp = F.resize_images(conv5_3_pool2_conv, (h,w))

        conv5_3_pool3 = F.average_pooling_2d(coonv5_3, h/3, stride=h/3)
        conv5_3_pool3_conv = self.bn5_3_pool3_conv(self.conv5_3_pool3_conv(conv5_3_pool3), test=test)
        conv5_3_pool3_interp = F.resize_images(conv5_3_pool3_conv, (h,w))

        conv5_3_pool6 = F.average_pooling_2d(coonv5_3, h/6, stride=h/6)
        conv5_3_pool6_conv = self.bn5_3_pool6_conv(self.conv5_3_pool6_conv(conv5_3_pool6), test=test)
        conv5_3_pool6_interp = F.resize_images(conv5_3_pool6_conv, (h,w))

        conv5_3_concat = F.concat((coonv5_3, conv5_3_pool6_interp))
        conv5_3_concat = F.concat((conv5_3_concat, conv5_3_pool3_interp))
        conv5_3_concat = F.concat((conv5_3_concat, conv5_3_pool2_interp))
        conv5_3_concat = F.concat((conv5_3_concat, conv5_3_pool1_interp))

        conv5_4 = F.relu(self.bn5_4(self.conv5_4(conv5_3_concat), test=test))
        conv5_4 = F.dropout(conv5_4, ratio=.1, train=train)

        conv6 = self.conv6(conv5_4)

        conv6_interp = F.resize_images(conv6, (self.input_height,self.input_width))

        return conv6_interp


class Network(BaseNetwork):
    def __init__(self,gpus,batchsize,data_dir,net,mode,epochs,save_every,size,lr, **kwargs):
        super(Network, self).__init__(epochs,save_every)
        print "building ..."
        self.input_height=size
        self.input_width=size
        self.net = net
        self.mode=mode
        self.train_data, self.test_data=self.get_dataset(data_dir)

        self.gpu1, self.gpu2, self.gpu3 = gpus
        self.pspnet = PSPNet(self.in_channel, self.out_channel,self.input_height,self.input_width,self.gpu1,self.gpu2,self.gpu3)
        self.pspnet.to_gpu(self.gpu1)

        self.xp = cuda.cupy
        cuda.get_device(self.gpu1).use()

        self.o_pspnet = optimizers.Adam(alpha=lr, beta1=0.5)
        self.o_pspnet.setup(self.pspnet)
        self.o_pspnet.add_hook(chainer.optimizer.WeightDecay(0.0001))

        self.batchsize=batchsize


    def read_batch(self, perm, batch_index,data_raw):
        data = np.zeros((self.batchsize, self.in_channel, self.input_height, self.input_width), dtype=np.float32)
        t = np.zeros((self.batchsize, self.input_height, self.input_width), dtype=np.int32)
        for j_,j in enumerate(perm[batch_index:batch_index+self.batchsize]):
            data[j_,:,:,:] = (data_raw[j][0]/127.5)-1
            t[j_,:,:] = data_raw[j][1]
        return data, t

    def step(self,perm,batch_index,mode,epoch):
        if mode=='train':
            data, t=self.read_batch(perm,batch_index,self.train_data) 
            data = Variable(cuda.to_gpu(data))
            t=Variable(cuda.to_gpu(t))

            y = self.pspnet(data, train=True, test=False)
            
            L_pspnet = F.softmax_cross_entropy(y, t)
            A_pspnet = F.accuracy(y, t, ignore_label=-1)

            self.pspnet.cleargrads()
            L_pspnet.backward()
            self.o_pspnet.update()


        else :
            data, t=self.read_batch(perm,batch_index,self.test_data)
            data = Variable(cuda.to_gpu(data))
            t=Variable(cuda.to_gpu(t))

            y = self.pspnet(data, train=False, test=True)
            
            L_pspnet = F.softmax_cross_entropy(y, t)
            A_pspnet = F.accuracy(y, t, ignore_label=-1)



        return {"prediction": y.data.get(),
                "current_loss": L_pspnet.data.get(),
                "current_accuracy": A_pspnet.data.get(),
        }

  
    def test(self):
        p = ProgressBar()
        sum_accuracy = np.float32(0)
        for i_  in p(range(0,len(self.test_data),self.batchsize)): 
            y2 = np.zeros((self.batchsize, self.in_channel, self.input_height, self.input_width), dtype=np.float32)
            t = np.zeros((self.batchsize, self.input_height, self.input_width), dtype=np.int32)
            for j in xrange(self.batchsize):
                y2[j,:,:,:] = (self.test_data[i_+j][0][:,:,:]/127.5)-1
                t[j,:,:] = self.test_data[i_+j][1]
            t=Variable(cuda.to_gpu(t, self.gpu1))
            y = self.pspnet(Variable(cuda.to_gpu(y2, self.gpu1)), test=True,train=False)
            sum_accuracy += F.accuracy(x, t, ignore_label=-1).data.get()
            y = y.data.get().transpose(0,2,3,1)
            for n in xrange(self.batchsize):            
                seg_image=np.argmax(y[n,:,:,:],axis=2)   
                color_image = self.palette[seg_image]
                cv2.imwrite('./test_images/%d_pspnet.png'%(self.test_label_path_list[i_+n]), color_image)
        print 'test accuracy = ', sum_accuracy/(len(self.test_data)/self.batchsize)


    def save_params(self, epoch):
        print "==> saving state %s" % self.out_model_dir
        serializers.save_hdf5("%s/model_pspnet_%d.h5"%(self.out_model_dir, epoch),self.pspnet)


    def load_state(self,path,epoch):
        print "==> loading state %s epoch %s"%(path,epoch)
        serializers.load_hdf5('./states/%s/model_pspnet_%s.h5'%(path,epoch), self.pspnet)

        return int(epoch)
