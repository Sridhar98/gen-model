from utils import *
from gcn import add_features
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
import dgl
import torch
from dgl import DGLGraph
from torchvision import transforms, utils
import json
from params import *


def data_augmentation(dir_path): # augments data and saves it
    #uses 75% as training, 10% for val and 15% for test sets respectively
    class_v = {}
    X_train = {}
    adj_train = {}
    for object_name in object_names:

        with open(dir_path+object_name+'_images', 'rb') as f:
          o_images = pickle.load(f)
          print(len(o_images))
        with open(dir_path+object_name+'_part_separated_labels', 'rb') as f:
          o_labels = pickle.load(f)
          print(len(o_labels))
        with open(dir_path+object_name+'_part_separated_masks', 'rb') as f:
          o_masks = pickle.load(f)
          print(len(o_masks))
        with open(dir_path+object_name+'_part_separated_bbx', 'rb') as f:
          o_bbx = pickle.load(f)
          print(len(o_bbx))

        train_set_limit = int(len(o_bbx)*(75/100))
        validation_set_limit = int(len(o_bbx)*(10/100))
        test_set_limit = int(len(o_bbx)*(15/100))

        print(object_name,train_set_limit,validation_set_limit,test_set_limit)

        label = o_labels[0:train_set_limit]
        box = o_bbx[0:train_set_limit]
        mask = o_masks[0:train_set_limit]
        image = o_images[0:train_set_limit]

        label_val = o_labels[train_set_limit:train_set_limit+validation_set_limit]
        box_val = o_bbx[train_set_limit:train_set_limit+validation_set_limit]
        mask_val = o_masks[train_set_limit:train_set_limit+validation_set_limit]
        image_val = o_images[train_set_limit:train_set_limit+validation_set_limit]

        label_test = o_labels[train_set_limit+validation_set_limit::]
        box_test = o_bbx[train_set_limit+validation_set_limit::]
        mask_test = o_masks[train_set_limit+validation_set_limit::]
        image_test = o_images[train_set_limit+validation_set_limit::]

        max_parts = len(label[0])

        #data augmentation starts ---------------------------------------------

        flipped_label = []
        flipped_box = []
        flipped_mask = []
        flipped_image = []

        label = label_test
        box = box_test
        mask = mask_test
        image = image_test

        print('At the beginning: ',len(box))

        angle = 3

        for l,b,m,i in zip(label,  box, mask, image):
            ll, bb, mm, ii = flip_data_instance(l,b,m,i,object_name)
            flipped_label.append(ll)
            flipped_box.append(bb)
            flipped_mask.append(mm)
            #flipped_image.append(ii)

        flipped_label = np.asarray(flipped_label)
        flipped_box =   np.asarray(flipped_box )
        flipped_mask =  np.asarray(flipped_mask)
        flipped_image = np.asarray(flipped_image)


        label = np.concatenate((label, flipped_label), axis = 0)
        box = np.concatenate((box, flipped_box), axis = 0)
        mask = np.concatenate((mask, flipped_mask), axis = 0)
        image = np.concatenate((image, flipped_image), axis = 0)

        print('After concatenating flipped ones: ',len(box))

        rt_box = []
        for bx, mx in zip(box, mask):
          bbxx1 = render_mask(bx,mx,angle)
          bbxx2 = render_mask(bx,mx,-angle)
          rt_box.append(bbxx1)
          rt_box.append(bbxx2)

        rt_box = np.asarray(rt_box)
        box = box = np.concatenate((box, rt_box), axis = 0)

        print('After concatenating rt: ', len(box))

        centre_box = []
        for bx in box:
          centre_box.append(centre_object(bx, (canvas_size,canvas_size)))

        centre_box = np.asarray(centre_box)
        box = centre_box

        scale_box = []
        for bx in box:
          a,b,c,d = scale(bx, 0.01)
          scale_box.append(a)
          scale_box.append(b)
          scale_box.append(c)
          scale_box.append(d)

        scale_box = np.asarray(scale_box)
        box = np.concatenate((box, scale_box), axis = 0)

        print('After concatenating scaled boxes: ',len(box))
        box = append_labels(box)
        box = pad_along_axis(box, 24, axis=1) #pads extra 0 rows at bottom
                                                # to get uniform size

        numparts = 24

        #data augmentation ends ---------------------------------------------

        final_input = []

        #generating the adjacency matrix

        for i,x in enumerate(box):
          l = numparts
          temp = np.zeros((l,l), dtype=np.float32)
          for j in range(l):
            temp[j][j] = 1
            if x[j][0] == 1:
              for y in tree[object_name][j]:
                if x[y][0] == 1:
                  temp[j][y] = 1
                  temp[y][j] = 1
          final_input.append(temp)
        adj = np.asarray(final_input)

        class_v[object_name] = (np.asarray([np.eye(10)[class_dic[object_name]]]*len(box)))
        X_train[object_name] = (box)
        adj_train[object_name] = (adj)
        print('len for ',object_name,len(X_train[object_name]))


    cclass_v = np.concatenate((
    class_v['cow'],
    class_v['person'],
    class_v['cat'],
    class_v['dog'],
    class_v['horse'],
    class_v['sheep'],
    class_v['bird'],
    class_v['aeroplane'],
    class_v['motorbike'],
    class_v['bicycle']), axis = 0)


    cadj_train = np.concatenate((
    adj_train['cow'],
    adj_train['person'],
    adj_train['cat'],
    adj_train['dog'],
    adj_train['horse'],
    adj_train['sheep'],
    adj_train['bird'],
    adj_train['aeroplane'],
    adj_train['motorbike'],
    adj_train['bicycle']), axis = 0)


    cX_train = np.concatenate((
    X_train['cow'],
    X_train['person'],
    X_train['cat'],
    X_train['dog'],
    X_train['horse'],
    X_train['sheep'],
    X_train['bird'],
    X_train['aeroplane'],
    X_train['motorbike'],
    X_train['bicycle']), axis = 0)


    X_train = cX_train
    class_v = cclass_v
    adj_train = cadj_train

    with open(dir_path+'X_test', 'wb') as fp:
        pickle.dump(X_train, fp)

    with open(dir_path+'class_v_test', 'wb') as fp:
        pickle.dump(class_v, fp)

    with open(dir_path+'adj_test', 'wb') as fp:
        pickle.dump(adj_train, fp)


def collate(samples):
    X,_,adj,y,pp = map(np.asarray,zip(*samples))
    X = np.reshape(X,[-1,5])
    print('In collate: X.shape',X.shape)
    pp = np.reshape(pp,[-1,parameters['parts']])
    adj = np.reshape(adj,[-1,parameters['parts'],parameters['parts']])
    _,A,_,_,_ = map(list,zip(*samples))
    A = dgl.batch(A)
    #X,A,y,pp = samples[0]

    return X,A,adj,y,pp


class GraphDataset_train(Dataset):

  def __init__(self,dir_path):

        #load train, val or test set from disk

        with open(dir_path+'X_train', 'rb') as fp:
            self.X_train = pickle.load(fp)
        with open(dir_path+'class_v', 'rb') as fp:
            self.class_v_train = pickle.load(fp)
        with open(dir_path+'adj_train', 'rb') as fp:
            self.adj_train = pickle.load(fp)




  def __len__(self):
        return len(self.X_train)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample

        # Load data and get label
        self.X = self.X_train[index]
        self.A = dgl.DGLGraph(self.adj_train[index])
        self.A = add_self_loop(self.A)
        #self.embed = add_features(self.A,parameters['parts'],self.X)
        self.pp = get_pos(self.X)
        self.y = self.class_v_train[index]

        return self.X,self.A,self.adj_train[index],self.y,self.pp

class GraphDataset_val(Dataset):

  def __init__(self,dir_path):

        #load train, val or test set from disk

        with open(dir_path+'X_val', 'rb') as fp:
            self.X_val = pickle.load(fp)
        with open(dir_path+'class_v_val', 'rb') as fp:
            self.class_v_val = pickle.load(fp)
        with open(dir_path+'adj_val', 'rb') as fp:
            self.adj_val = pickle.load(fp)




  def __len__(self):
        return len(self.X_val)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample

        # Load data and get label
        self.X = self.X_val[index]
        self.A = dgl.DGLGraph(self.adj_val[index])
        self.A = add_self_loop(self.A)
        #self.embed = add_features(self.A,parameters['parts'],self.X)
        self.pp = get_pos(self.X)
        self.y = self.class_v_val[index]

        return self.X,self.A,self.adj_val[index],self.y,self.pp

class GraphDataset_test(Dataset):

  def __init__(self,dir_path):

        #load train, val or test set from disk


        with open(dir_path+'X_test', 'rb') as fp:
            self.X_test = pickle.load(fp)
        with open(dir_path+'class_v_test', 'rb') as fp:
            self.class_v_test = pickle.load(fp)
        with open(dir_path+'adj_test', 'rb') as fp:
            self.adj_test = pickle.load(fp)




  def __len__(self):
        return len(self.X_test)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample

        # Load data and get label
        self.X = self.X_test[index]
        self.A = dgl.DGLGraph(self.adj_test[index])
        self.A = add_self_loop(self.A)
        #self.embed = add_features(self.A,parameters['parts'],self.X)
        self.pp = get_pos(self.X)
        self.y = self.class_v_test[index]

        return self.X,self.A,self.adj_test[index],self.y,self.pp
