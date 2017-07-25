from chainer import serializers
from chainer import Variable
import numpy as np
import chainer.functions as F
from chainer import cuda
import os
import cv2
from PIL import Image
import json

class BaseNetwork(object):

    def __init__(self, epochs, save_every):
        self.save_every = save_every
        self.epochs=epochs
    
    def my_state(self):
        return '%s'%(self.net)
    
    def save_params(self, epoch):
        print "==> saving state %s" % self.out_model_dir
        serializers.save_hdf5("%s/net_model_classifier_%d.h5"%(self.out_model_dir, epoch),self.network)
    

    def load_state(self,path,epoch):
        print "==> loading state %s epoch %s"%(path,epoch)
        serializers.load_hdf5('./states/%s/net_model_classifier_%s.h5'%(path,epoch), self.network)
        return int(epoch)


    def read_batch(self, perm, batch_index, data_raw):

        data = np.zeros((self.batchsize, self.in_channel, self.input_height, self.input_width), dtype=np.float32)
        label = np.zeros((self.batchsize), dtype=np.int32)

        for j_,j in enumerate(perm[batch_index:batch_index+self.batchsize]):
                data[j_,:,:,:] = data_raw[j][0].astype(np.float32)
                label[j_] = int(data_raw[j][1])
        return data, label
    
    
    def step(self,perm,batch_index, mode, epoch): 
            if mode =='train':
                data, label=self.read_batch(perm,batch_index,self.train_data)
            else:
                data, label=self.read_batch(perm,batch_index,self.test_data)

            data = Variable(cuda.to_gpu(data))
            yl = self.network(data)

            label=Variable(cuda.to_gpu(label))

            L_network = F.softmax_cross_entropy(yl, label)
            A_network = F.accuracy(yl, label)

            if mode=='train':
                self.o_network.zero_grads()
                L_network.backward()
                self.o_network.update()


            return {"prediction": yl.data.get(),
                    "current_loss": L_network.data.get(),
                    "current_accuracy": A_network.data.get(),
            }

  
    def get_dataset(self, data_dir):

        with open('./utils/cityscape.json', 'r') as f:
            json_data = json.load(f)
        self.palette = np.array(json_data['palette'], dtype=np.uint8)

        train_image_dir = data_dir+'/leftImg8bit/train/'
        train_label_dir = data_dir+'/gtFine/train/'
        test_image_dir = data_dir+'/leftImg8bit/val/'
        test_label_dir = data_dir+'/gtFine/val/'
        self.out_channel=19
        self.in_channel=3

        dst='labelIds.png'
        self.test_image_path_list=[]
        self.test_label_path_list=[]

        train_label_path_list=[]
        train_label_city_list=os.listdir(train_label_dir)
        train_label_city_list.sort()
        if self.mode =='train':
            for city in train_label_city_list:
                namelist=os.listdir(train_label_dir+city)
                namelist.sort()
                list_tmp=[train_label_dir+city+'/'+name for name in namelist if name.endswith(dst)]
                train_label_path_list.append(list_tmp)

        test_label_path_list=[]
        test_label_city_list=os.listdir(test_label_dir)
        test_label_city_list.sort()
        for city in test_label_city_list:
            namelist=os.listdir(test_label_dir+city)
            namelist.sort()
            list_tmp=[test_label_dir+city+'/'+name for name in namelist if name.endswith(dst)]
            test_label_path_list.append(list_tmp)

        train_image_path_list=[]
        train_image_city_list=os.listdir(train_image_dir)
        train_image_city_list.sort()
        if self.mode =='train':
            for city in train_image_city_list:
                namelist=os.listdir(train_image_dir+city)
                namelist.sort()
                list_tmp=[train_image_dir+city+'/'+name for name in namelist]
                train_image_path_list.append(list_tmp)

        test_image_path_list=[]
        test_image_city_list=os.listdir(test_image_dir)
        test_image_city_list.sort()
        for city in test_image_city_list:
            namelist=os.listdir(test_image_dir+city)
            namelist.sort()
            list_tmp=[test_image_dir+city+'/'+name for name in namelist]
            test_image_path_list.append(list_tmp)

        train_data_X=[]
        test_data_X=[]
        train_label_dataset=[]
        self.test_label_dataset=[]
        
        if self.mode =='train':
            for fn in train_image_path_list:
                for fi in fn:
                    train_data_X.append(np.asarray(Image.open(fi).resize((self.input_width, self.input_height)).convert('RGB')).astype(np.float32).transpose(2, 0, 1))
                    
        for fn in test_image_path_list:
            for fi in fn: 
                test_data_X.append(np.asarray(Image.open(fi).resize((self.input_width, self.input_height)).convert('RGB')).astype(np.float32).transpose(2, 0, 1))
                self.test_image_path_list.append(fi)
        
        label_dic = {0:-1, 1:-1, 2:-1, 3:-1, 4:-1, 5:-1, 6:-1, 7:0, 8:1, 9:-1, 10:-1, 11:2, 12:3, 13:4, 14:-1, 15:-1, 16:-1, 17:5, 18:-1, 19:6, 20:7, 21:8, 22:9, 23:10, 24:11, 25:12, 26:13, 27:14, 28:15, 29:-1, 30:-1, 31:16, 32:17, 33:18}        
        
        if self.mode =='train':
            for fn in train_label_path_list:
                for fi in fn:
                    if fi.endswith(dst):
                        img = np.asarray(Image.open(fi).resize((self.input_width, self.input_height)).convert('RGB')).astype(np.int32).transpose(2, 0, 1)
                        img2 = np.zeros((self.input_height, self.input_width),dtype=np.int32)
                        for i in range(34):
                            bool_map = img[0,:,:]==i
                            img2 += bool_map*label_dic[i]
                        train_label_dataset.append(img2)

        for fn in test_label_path_list:
            for fi in fn:
                if fi.endswith(dst):
                    img = np.asarray(Image.open(fi).resize((self.input_width, self.input_height)).convert('RGB')).astype(np.int32).transpose(2, 0, 1)
                    self.test_label_path_list.append(fi)
                    _,h,w=img.shape
                    img2 = np.zeros((self.input_height, self.input_width),dtype=np.int32)
                    for i in range(34):
                        bool_map = img[0,:,:]==i
                        img2 += bool_map*label_dic[i]
                    self.test_label_dataset.append(img2) 


        self.out_model_dir ='./states/'+self.my_state()

        if not os.path.exists(self.out_model_dir):
            os.makedirs(self.out_model_dir)

        if self.mode=='train':
            print "==> %d training examples" % len(train_data_X)
            print "out_model_dir ==> %s " % self.out_model_dir
            print "==> %d test examples" % len(test_data_X)
        else:
            print "==> %d test examples" % len(test_data_X)


        train_data=[]
        for i,j in zip(train_data_X,train_label_dataset):
            train_data.append((i,j))

        test_data=[]
        for i,j in zip(test_data_X,self.test_label_dataset):
            test_data.append((i,j))


        return train_data, test_data
