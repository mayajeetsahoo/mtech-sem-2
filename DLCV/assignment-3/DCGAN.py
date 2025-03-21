import argparse
import os
import numpy as np
import math
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
import random
from torch.utils.data import Dataset,DataLoader
import matplotlib.pyplot as plt

device = "cuda:0" if torch.cuda.is_available() else "cpu"

latent_dim = 100

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.img_dim_in_gen = 32//4
        self.init_channel = 128
        self.compression_factor = 0.5
        self.linear = nn.Linear(latent_dim,self.init_channel*self.img_dim_in_gen**2)
        self.conv_block = nn.Sequential(nn.BatchNorm2d(self.init_channel),
                                        
                                        nn.Upsample(scale_factor=2),
                                        nn.Conv2d(self.init_channel,int(self.init_channel*self.compression_factor),3, stride=1, padding=1),

                                        nn.BatchNorm2d(int(self.init_channel*self.compression_factor)),
                                        nn.LeakyReLU(0.2,inplace=True),

                                        nn.Upsample(scale_factor=2),
                                        nn.Conv2d(int(self.init_channel*self.compression_factor),int(self.init_channel*self.compression_factor*self.compression_factor),3, stride=1, padding=1),

                                        nn.BatchNorm2d(int(self.init_channel*self.compression_factor*self.compression_factor)),
                                        nn.LeakyReLU(0.2,inplace=True),

                                        nn.Conv2d(int(self.init_channel*self.compression_factor*self.compression_factor),3,3, stride=1, padding=1),
                                        nn.Tanh()

                                        )
    
    def forward(self,input):
        out_linear = self.linear(input)
        out_projection_reshape = out_linear.view(out_linear.shape[0],self.init_channel,self.img_dim_in_gen,self.img_dim_in_gen)
        out = self.conv_block(out_projection_reshape)
        return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv_block = nn.Sequential(nn.Conv2d(3,16,3,2,1),
                                        nn.LeakyReLU(0.2,inplace=True),nn.Dropout(0.25),
            nn.BatchNorm2d(16),
            nn.Conv2d(16,32,3,2,1),nn.LeakyReLU(0.2,inplace=True),nn.Dropout(0.25),nn.BatchNorm2d(32))
        final_flat_size = 32*(32//2**2)**2
        self.linear = nn.Linear(final_flat_size,1)
        self.final_activation = nn.Sigmoid()
    def forward(self,image):
        out_conv = self.conv_block(image)
        flat = out_conv.view(out_conv.shape[0],-1)
        logits = self.linear(flat)
        logits=self.final_activation(logits)
        return logits
    


class MyDataset(Dataset):
    def __init__(self, trainset, indices, transforms):
        self.cifar10_dataset = trainset
        self.indices = indices
        self.transforms = transforms

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        pil_img, label = self.cifar10_dataset[self.indices[idx]]
        img = self.transforms(pil_img)
        return img, label

class get_data_loaders():
    def __init__(self,batch_size,percentage=0.9,low_percentage=1):
        transform = transforms.Compose([transforms.ToTensor()])
        # Load CIFAR-10 training set
        train_indices = random.sample(range(0, 50000), int(percentage*50000))
        val_indices = list(set(range(50000))-set(train_indices))
        train_indices_low = random.sample(train_indices,int(low_percentage*len(train_indices)))
        self.percentage=percentage
        self.batch_size=batch_size
        self.trainset = torchvision.datasets.CIFAR10(
            root='./data',
            train=True,
            download=True
            # ,transform=transform
        )
        self.testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True,transform=transform
        )
        self.testset_object = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True
        )
        trainset_custom=MyDataset(self.trainset,train_indices_low,transform)
        validation_custom=MyDataset(self.trainset,val_indices,transform)
        self.validationloader = torch.utils.data.DataLoader(validation_custom, batch_size=self.batch_size, shuffle=False)
        self.trainloader = torch.utils.data.DataLoader(trainset_custom, batch_size=self.batch_size, shuffle=True)

generator_model = Generator()
discriminator_model = Discriminator()

loss = torch.nn.BCELoss()

gen_optimizer = torch.optim.SGD(generator_model.parameters(),lr = 0.0002)
disc_optimizer = torch.optim.SGD(discriminator_model.parameters(),lr = 0.0002)


generator_model.to(device)
discriminator_model.to(device)
loss.to(device)

# # Initialize weights
# generator_model.apply(weights_init_normal)
# discriminator_model.apply(weights_init_normal)



## get trainloader and test loaders
# data = get_data_loaders(128)

# epochs = 1

# epochwise_d_loss = []
# epochwise_g_loss = []

# all_d_loss=[]
# all_g_loss=[]
# for epoch in range(epochs):
#     print(epoch)
#     gen_loss=[]
#     disc_loss=[]
#     for batch_idx,(images,_) in enumerate(data.trainloader):
#         # print(images.shape)
#         rand_dis = Variable(torch.FloatTensor(np.random.normal(0,1,(images.shape[0],latent_dim))), requires_grad=False).to(device)
#         real_images = images.to(device)
#         real_labels = torch.ones(real_images.shape[0],1).type(torch.FloatTensor).to(device)

#         fake_images = generator_model(rand_dis)
#         fake_labels = torch.zeros(fake_images.shape[0],1).type(torch.FloatTensor).to(device)
        
#         # real_labels = Variable(torch.FloatTensor(real_images.shape[0], 1).fill_(1.0), requires_grad=False).to(device)
#         # fake_labels = Variable(torch.FloatTensor(fake_images.shape[0], 1).fill_(0.0), requires_grad=False).to(device)


#         ## training generator

#         # Loss measures generator's ability to fool the discriminator

#         g_loss = loss(discriminator_model(fake_images), real_labels)
#         gen_loss.append(g_loss.item())
#         all_g_loss.append(g_loss.item())
        
#         gen_optimizer.zero_grad()
#         g_loss.backward()
#         gen_optimizer.step()


#         ## training discriminator
#         # Measure discriminator's ability to classify real from generated samples
#         real_loss = loss(discriminator_model(real_images), real_labels)
#         fake_loss = loss(discriminator_model(fake_images.detach()), fake_labels)
#         d_loss = (real_loss + fake_loss) / 2
#         disc_loss.append(d_loss.item())
#         all_d_loss.append(d_loss.item())
#         d_loss.backward()
#         disc_optimizer.step()


#         disc_optimizer.zero_grad()
    
#     epochwise_d_loss.append(np.mean(disc_loss))
#     epochwise_g_loss.append(np.mean(gen_loss))

        

# plt.plot(range(epochs),epochwise_d_loss,label="discriminator loss")
# plt.plot(range(epochs),epochwise_g_loss,label="generator loss")
# plt.legend()
# plt.savefig("plots/question1/gan.png")
# plt.close()
# plt.plot(range(len(all_d_loss)),all_d_loss,label="discriminator loss")
# plt.plot(range(len(all_g_loss)),all_g_loss,label="generator loss")
# plt.legend()
# plt.savefig("plots/question1/gan_all.png")
# plt.close()




noise=torch.FloatTensor(np.random.normal(0,1,(1,latent_dim))).to(device)



generator_model.load_state_dict(torch.load("gen_model.pt",map_location=torch.device('cpu')))


gen_img = generator_model(noise)
gen_img=gen_img.detach().numpy()


plt.imshow((np.squeeze(gen_img,axis=0).T*255).astype(np.uint8))
plt.show()