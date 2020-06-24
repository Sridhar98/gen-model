import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import json
from params import *



class CVAE(nn.Module):
    def __init__(self, feature_size, latent_size, class_size):
        super(CVAE, self).__init__()
        self.feature_size = feature_size
        self.class_size = class_size

        # encode
        self.fc1  = nn.Linear(parameters['parts']*16,128)
        self.fc2  = nn.Linear(128,128)
        self.fc3  = nn.Linear(feature_size + class_size, 128)
        self.fc41 = nn.Linear(128, latent_size)
        self.fc42 = nn.Linear(128, latent_size)

        # decode
        self.fc5 = nn.Linear(parameters['latent_size']+parameters['parts']+parameters['class_size'], 128)
        self.fc6 = nn.Linear(128, 128)
        self.fc7 = nn.Linear(128, 128)
        self.fc81 = nn.Linear(128,4*parameters['parts']) #bbox sub-matrix
        self.fc82 = nn.Linear(128,1*parameters['parts']) #part presence vec
        self.fc83 = nn.Linear(128,parameters['parts']*parameters['parts']) #adj matrix
        self.fc84 = nn.Linear(128,class_size)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def encode(self,inputs,skip,c,pp): # Q(z|x, c)
        '''
        x: feature_size
        c: class_size
        '''
        h1 = self.relu(self.fc1(inputs))
        h1 = h1 + skip  # skip connection
        h2 = self.relu(self.fc2(h1)) #float32
        h2 = torch.cat((c,h2),-1)
        h3 = self.relu(self.fc3(h2))
        z_mu = self.relu(self.fc41(h3))
        z_var = self.relu(self.fc42(h3))
        return z_mu, z_var

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z, cpp): # P(x|z, c)
        '''
        z: latent_size
        c: class_size
        '''
        z = torch.cat((cpp,z),-1)
        h5 = self.relu(self.fc5(z))
        h6 = self.relu(self.fc6(h5))
        h7 = self.relu(self.fc7(h6))
        bbox = self.sigmoid(self.fc81(h7))
        part_vec = self.sigmoid(self.fc82(h7))
        A = self.sigmoid(self.fc83(h7))
        class_ = self.softmax(self.fc84(h7))

        return bbox,part_vec,A,class_

    def forward(self, inputs, skip, c, pp):
        c = c.type('torch.FloatTensor')
        pp = pp.type('torch.FloatTensor')
        mu, logvar = self.encode(inputs, skip, c,pp)
        z = self.reparameterize(mu, logvar)
        cpp = torch.cat((c,pp),-1)
        bbox, part_vec, A, class_ = self.decode(z, cpp)
        return bbox, part_vec, A, class_, mu, logvar, z, cpp
