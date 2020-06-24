import torch
import torch.nn as nn
import numpy as np
import argparse, time
import networkx as nx
import torch.nn.functional as F
from dgl.data import register_data_args, load_data
from dgl.nn.pytorch import GraphConv
from dgl import DGLGraph
from gcn import *
from cvae_box import *
from data_utils import *
from params import *
from IPython.display import clear_output
import matplotlib.pyplot as plt

def main(args):

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    #augment dataset
    #data_augmentation('/home/sridhar/IIITH/Dataset/data/bin_data/')

    #open log file
    f = open(args.ls_path+'log.txt',w+)

    #create dataset
    train_dataset = GraphDataset_train(dir_path)
    val_dataset = GraphDataset_val(dir_path)
    #test_dataset = GraphDataset_test(dir_path)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=parameters['batch_size'],shuffle=True,num_workers=0,collate_fn=collate)
    val_dataloader = torch.utils.data.DataLoader(val_dataset,batch_size=parameters['batch_size'],shuffle=True,num_workers=0,collate_fn=collate)

    # Construct GCN
    gmodel = GCN(5, 32, 16)

    #Initialize CVAE
    cvae = CVAE(2*parameters['latent_size'],parameters['latent_size'],parameters['class_size'])

    #Initialize optimizer
    optimizer = torch.optim.Adam(list(gmodel.parameters())+list(cvae.parameters()),lr=0.0001)

    #if resuming from checkpoint, load the model and optimizer from checkpoint file
    #set epoch start from file and start of frange cycle icoef
    if args.resume:
        checkpoint = torch.load(args.ls_path)
        gmodel.load_state_dict(checkpoint['GCN_state_dict'])
        cvae.load_state_dict(checkpoint['CVAE_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_start = checkpoint['epoch']
    else:
        epoch_start = 1

    #frange_cycle coefficient list
    klw = frange_cycle_linear(10000)


    icoef = epoch_start #idx into klw coeff list

    #init lists to store loss/epoch history for visualization
    train_loss = []
    val_loss = []

    for i in range(epoch_start,args.n_epochs+1):

        #sum of training and val losses
        r_loss = 0
        r_loss_val = 0

        for X,G,ADJ,y,pp in train_dataloader:


            #get adjacency matrix of graph A
            adj = dgl.DGLGraph.adjacency_matrix(G)
            adj = adj.to(device)

            add_features(G,parameters['parts'],X)

            X = torch.from_numpy(X)
            X = X.to(device)

            batch_size = X.shape[0] // parameters['parts']
            print('Batch size: ',batch_size)

            y = y.astype(float)
            y = torch.from_numpy(y)
            y = y.type('torch.LongTensor')

            #converting one-hot class vec to class indices
            y_new = torch.zeros(batch_size)
            for i in range(batch_size):
                y_new[i] = torch.argmax(y[i,:],axis=0)
            y_new = y_new.type('torch.LongTensor')
            y_new = y_new.to(device)

            pp = torch.from_numpy(pp)
            pp = pp.type('torch.FloatTensor')
            pp = pp.to(device)

            ADJ = torch.from_numpy(ADJ)
            ADJ = ADJ.to(device)

            #get inputs
            inputs = X.float()

            #GCN forward pass
            h, skip = gmodel(G,inputs,adj)
            h = torch.reshape(h,(-1,parameters['parts']*16))

            #VAE forward pass
            bbox,part_vec,A,class_,mu, logvar, z, cpp = cvae(h,skip,y,pp)
            bbox = torch.reshape(bbox,(-1,parameters['parts'],4))
            A = torch.reshape(A,(-1,parameters['parts'],parameters['parts']))
            A = A.to(device)

            #Compute training loss
            X = torch.reshape(X,(-1,parameters['parts'],5))
            X = X.type('torch.FloatTensor')
            bbox_loss = compute_ciou(X[:,:,1:],bbox) + box_loss(X[:,:,1:],bbox)
            kl_loss = torch.mean(torch.sum(mu**2 + torch.exp(logvar)**2 - 2*logvar - 1,axis=1))
            bce_loss = torch.nn.BCELoss()
            categoricalCE_loss = torch.nn.CrossEntropyLoss()
            adj_loss = torch.mean(bce_loss(A,ADJ))
            class_ = class_.to(device)
            cls_loss = torch.mean(categoricalCE_loss(class_,y_new))
            part_vec = part_vec.to(device)
            pp_loss = torch.mean(bce_loss(part_vec,pp))
            reconstruction_loss = (bbox_loss + cls_loss + adj_loss + pp_loss)*24*5 + klw[icoef]*kl_loss
            r_loss = r_loss + reconstruction_loss
            break
            batch = batch + 1
            #clear previous gradients
            optimizer.zero_grad()

            #Backpropagation
            reconstruction_loss.backward()

            optimizer.step()

        r_loss = r_loss/2512 #avg train loss per epoch
        train_loss.append(r_loss)

        with torch.no_grad():
            batch_val = 0
            for X_val,G_val,ADJ_val,y_val,pp_val in val_dataloader:

                #get adjacency matrix of graph A
                adj_val = dgl.DGLGraph.adjacency_matrix(G_val)
                adj_val = adj_val.to(device)

                add_features(G_val,parameters['parts'],X_val)

                X_val = torch.from_numpy(X_val)
                X_val = X_val.to(device)

                batch_size_val = X_val.shape[0] // parameters['parts']

                y_val = y_val.astype(float)
                y_val = torch.from_numpy(y_val)
                y_val = y_val.type('torch.LongTensor')

                #converting one-hot class vec to class indices
                y_new_val = torch.zeros(batch_size_val)
                for i in range(batch_size_val):
                    y_new_val[i] = torch.argmax(y_val[i,:],axis=0)
                y_new_val = y_new_val.type('torch.LongTensor')
                y_new_val = y_new_val.to(device)

                pp_val = torch.from_numpy(pp_val)
                pp_val = pp_val.type('torch.FloatTensor')
                pp_val = pp_val.to(device)

                ADJ_val = torch.from_numpy(ADJ_val)
                ADJ_val = ADJ_val.to(device)

                #get inputs
                inputs_val = X_val.float()

                #GCN forward pass
                h_val, skip_val = gmodel(G_val,inputs_val,adj_val)
                h_val = torch.reshape(h_val,(-1,parameters['parts']*16))

                #VAE forward pass
                bbox_val,part_vec_val,A_val,class__val,mu_val, logvar_val, z_val, cpp_val = cvae(h_val,skip_val,y_val,pp_val)
                bbox_val = torch.reshape(bbox_val,(-1,parameters['parts'],4))
                A_val = torch.reshape(A_val,(-1,parameters['parts'],parameters['parts']))
                A_val = A_val.to(device)

                #Compute val loss
                X_val = torch.reshape(X_val,(-1,parameters['parts'],5))
                X_val = X_val.type('torch.FloatTensor')
                bbox_loss_val = compute_ciou(X_val[:,:,1:],bbox_val) + box_loss(X_val[:,:,1:],bbox_val)
                kl_loss_val = torch.mean(torch.sum(mu_val**2 + torch.exp(logvar_val)**2 - 2*logvar_val - 1,axis=1))
                bce_loss_val = torch.nn.BCELoss()
                categoricalCE_loss_val = torch.nn.CrossEntropyLoss()
                adj_loss_val = torch.mean(bce_loss(A_val,ADJ_val))
                class__val = class__val.to(device)
                cls_loss_val = torch.mean(categoricalCE_loss(class__val,y_new_val))
                part_vec_val = part_vec_val.to(device)
                pp_loss_val = torch.mean(bce_loss(part_vec_val,pp_val))
                reconstruction_loss_val = (bbox_loss_val + cls_loss_val + adj_loss_val + pp_loss_val)*24*5 + klw[icoef]*kl_loss_val
                r_loss_val = r_loss_val + reconstruction_loss_val
                batch_val = batch_val + 1
                #break

        r_loss_val = r_loss_val/332
        val_loss.append(r_loss_val)

        clear_output()

        #print train, val loss after every epoch
        print('Epoch ',i)
        print('Training loss:',r_loss,'bbox_loss:',bbox_loss,'cls_loss:',cls_loss,'adj_loss:',adj_loss,'pp_loss:',pp_loss,'kl_loss:',kl_loss)
        print('Validation loss:',r_loss_val,'bbox_loss_val:',bbox_loss_val,'cls_loss_val:',cls_loss_val,'adj_loss_val:',adj_loss_val,'pp_loss_val:',pp_loss_val,'kl_loss_val:',kl_loss_val)
        print('KL coeff:',klw[icoef])

        #write loss info into log file
        f.write('Epoch '+i+'\n')
        f.write('Training loss: '+r_loss+'bbox_loss: '+bbox_loss+'cls_loss: '+cls_loss+'adj_loss: '+adj_loss+'pp_loss: '+pp_loss+'kl_loss: '+kl_loss+'\n')
        f.write('Validation loss: '+r_loss_val+'bbox_loss_val: '+bbox_loss_val+'cls_loss_val: '+cls_loss_val+'adj_loss_val: '+adj_loss_val+'pp_loss_val: '+pp_loss_val+'kl_loss_val: '+kl_loss_val+'\n')
        f.write('KL coeff: '+klw[icoef]+'\n')
        f.write('-----------------------------'+'\n')


        #freeze klw schedule if loss diff above threshold
        if kl_loss>0.5 and abs(r_loss - r_loss_val) < 0.2:
            icoef = icoef + 1

        #plot of train,val loss vs epochs
        if i%10 == 0:
            plt.plot(np.asarray(train_loss))
            plt.plot(np.asarray(val_loss))
            plt.show()


        #save model and optimizer every x epochs - checkpoint
        if i % args.save_every == 0:
            torch.save({
            'epoch': i+1,
            'GCN_state_dict': gmodel.state_dict(),
            'CVAE_state_dict': cvae.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
            }, args.ls_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    register_data_args(parser)
    parser.add_argument("--gpu", type=int, default=-1,
            help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
            help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=200,
            help="number of training epochs
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--ls-path", type=str)
    parser.add_argument("--checkpoint-fname", type=str)
    parser.add_argument("--weight-decay", type=float, default=5e-4,
            help="Weight for L2 loss")
    parser.add_argument("--resume",action='store_true',
            help="Resuming training from a checkpoint")
    parser.add_argument("--self-loop", action='store_true',
            help="graph self-loop (default=True)")
    parser.set_defaults(self_loop=True)
    args = parser.parse_args()
    print(args)

    main(args)
