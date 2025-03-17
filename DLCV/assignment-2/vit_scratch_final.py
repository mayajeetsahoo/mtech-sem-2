import torch
import numpy as np
from torch import nn 
import random
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.utils.data import Dataset,DataLoader
import cv2

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device="cpu"
def set_seed(seed: int, device=torch.device("cuda")) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available() and device == torch.device("mps"):
        pass
        # torch.cuda.manual_seed(seed)
        # torch.backends.cudnn.benchmark = False # type: ignore
        # torch.backends.cudnn.deterministic = True # type: ignore
        # torch.mps.manual_seed(seed)  # MPS-specific seed setting


class VitImageEmbeddings(nn.Module):
    def __init__(self, in_channel, patch_size, input_dim, height , width , stride_):
        super(VitImageEmbeddings,self).__init__()
        self.projection = nn.Conv2d(in_channel, input_dim, kernel_size=patch_size, stride=stride_, bias=False)
        self.out_height = ((height-patch_size)/stride_) + 1
        self.out_width = ((width-patch_size)/stride_) + 1
        self.sequence_length = int(self.out_height * self.out_width)

    def forward(self, image):
        """arguments:
                    image - input image is of size (batch_size, channels, height, width)
                    patch_size - size of patch we need from the image to be treated as a single sequence
                    input_dim - size of embedding vector
                    overlap - overlapping between patches of sequence
            returns:
                    embedding vector of each image .
                    output size - (batch_size, sequence_length, embedding dimension)"""
        batch_size, in_channel, height , width = image.shape
        embeddings = self.projection(image).flatten(2).transpose(1,2)
        return embeddings


class VitEmbedding(nn.Module):
    def __init__(self,in_channel, patch_size, input_dim, height , width , stride_):
        super().__init__()
        self.clstoken=nn.Parameter(torch.randn(1,1,input_dim))
        self.imgemb=VitImageEmbeddings(in_channel, patch_size, input_dim, height , width , stride_)
        # self.position=nn.Parameter(torch.randn(1,self.imgemb.sequence_length+1,input_dim))
        self.position=nn.Embedding(self.imgemb.sequence_length+1,input_dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self,image):
        """arguments:
                    image - input image is of size (batch_size, channels, height, width)
                    patch_size - size of patch we need from the image to be treated as a single sequence
                    input_dim - size of embedding vector
                    overlap - overlapping between patches of sequence
            returns:
                    embedding vector of each image concateneated with CLS token and positional embeddings.
                    output size - (batch_size, sequence_length, embedding dimension)"""
        batch_size, in_channel, height , width = image.shape
        embeddings=self.imgemb(image)
        cls=self.clstoken.expand(batch_size,-1,-1)
        embeddings_=torch.cat((cls,embeddings),dim=1)
        # position
        in_=torch.unsqueeze(torch.tensor(range(embeddings_.shape[1])),dim=0)
        in__=in_.expand(batch_size,-1).to(device)
        embs_=self.position(in__)
        added_positional_emb=embeddings_+embs_
        # added_positional_emb=embeddings_+self.position
        # added_positional_emb = self.dropout(added_positional_emb)
        return added_positional_emb


class LayerNorm(nn.Module):
    def __init__(self,input_dim):
        super().__init__()
        self.layer_norm=nn.LayerNorm(input_dim)

    def forward(self,embeddings):
        """arguments:
                    image - embeddings of shape (batch_size, sequence_length, embedding dimension)
            returns:
                    normalize each layer of embedding dimension.
                    output size - (batch_size, sequence_length, embedding dimension)"""

        normed_output=self.layer_norm(embeddings)
        return normed_output


    
class multimodal_self_attention_opt(nn.Module):
    def __init__(self,num_attention_heads,input_dim):
        super().__init__()
        self.num_attention_heads=num_attention_heads
        self.input_dim=input_dim
        self.Q=nn.Linear(input_dim,input_dim,bias=False)
        self.K=nn.Linear(input_dim,input_dim,bias=False)
        self.V=nn.Linear(input_dim,input_dim,bias=False)
        self.att_dim=int(self.input_dim/self.num_attention_heads)
    def size_change(self,tensor):
        new_tensor_shape = tensor.size()[:-1] + (self.num_attention_heads, self.att_dim)
        tensor = tensor.view(new_tensor_shape)
        return tensor.permute(0, 2, 1, 3)

    def forward(self,input):
        """arguments:
                    input - embedding matrix of shape (batch_size, sequence_length, embedding dimension)
                    num_attention_heads - number of attention head used for multimodal self attention
            returns:
                    self attention is applied across the input refer to attention is all you need paper for more details.
                    output size - (batch_size, sequence_length, embedding dimension)"""
        queries = self.size_change(self.Q(input))
        keys = self.size_change(self.K(input))
        values = self.size_change(self.V(input))

        dots_products = torch.matmul(queries,keys.transpose(-2,-1))
        dots_products = dots_products / np.sqrt(self.att_dim)
        soft_dot_out = nn.functional.softmax(dots_products,dim=-1)
        attentions = torch.matmul(soft_dot_out,values)
        ## implemented 2 codes for calculating self attention above and below gives the same output

        # dots_products = torch.matmul(keys,queries.transpose(-2,-1))
        # dots_products = dots_products / np.sqrt(self.att_dim)
        # soft_dot_out = nn.functional.softmax(dots_products,dim=-2)
        # attentions = torch.matmul(values.transpose(-2,-1),soft_dot_out)
        # attentions = attentions.transpose(-2,-1)

        attentions = attentions.permute(0, 2, 1, 3).contiguous()
        new_attention_layer_shape = attentions.size()[:-2] + (self.input_dim,)
        attentions = attentions.view(new_attention_layer_shape)
        return attentions,soft_dot_out



class MLP(nn.Module):
    def __init__(self,input_dim,hid,out_dim):
        super().__init__()
        self.lin1=nn.Linear(input_dim,hid)
        self.non_lin=nn.ReLU()
        self.lin2=nn.Linear(hid,out_dim)

    def forward(self,input):
        mlp_out_dummy = self.lin1(input)
        mlp_out_non=self.non_lin(mlp_out_dummy)
        mlp_out=self.lin2(mlp_out_non)

        return mlp_out


class VIT_encoder(nn.Module):
    def __init__(self,channels,patch_size,emb_dim,hid,height,width,stride,num_att_head):
        super().__init__()
        self.norm1=LayerNorm(emb_dim)
        self.att=multimodal_self_attention_opt(num_att_head,emb_dim)
        self.norm2=LayerNorm(emb_dim)
        self.mlp1=MLP(emb_dim,hid,emb_dim)

    def forward(self,trans_in):
        """arguments:
                    trans_in - embeddings of shape (batch_size, sequence_length, embedding dimension)
            returns:
                    the output of single VIT encoder.
                    output size - (batch_size, sequence_length, embedding dimension)"""
        lm1=self.norm1(trans_in)
        att_out = self.att(lm1)
        att_prev=att_out[1]
        att_=att_out[0]
        add1=trans_in+att_
        lm2=self.norm2(add1)
        mlp_=self.mlp1(lm2)
        add2=add1+mlp_
        return lm1, att_, add1, lm2, mlp_, add2, att_prev
        


class VIT_classification(nn.Module):
    def __init__(self,num_classes,channels,patch_size,emb_dim,hid,height,width,stride,num_att_head,num_encoder=1):
        super().__init__()
        self.num_encoder=num_encoder
        self.ve=VitEmbedding(channels,patch_size,emb_dim,height,width,stride)
        self.vit=VIT_encoder(channels,patch_size,emb_dim,hid,height,width,stride,num_att_head)
        self.vit_multi=nn.ModuleList([self.vit for i in range(self.num_encoder)])
        self.linear=nn.Linear(emb_dim,num_classes)
    def forward(self,image):
        """arguments:
                    image - takes image (batch_size, chanel, height , width)
            returns:
                    The logits of no of classes to predict.
                    output size - (batch_size, number of classses)"""
        transformer_input=self.ve(image)
        ## below block of code returns the output of different layer of VIT encoder
        self.qxp5=[]
        self.qxp6=[]
        enout=self.vit_multi[0](transformer_input)
        self.qxp5.append(enout[5])
        self.qxp6.append(enout[6])
        if self.num_encoder>1:
            for encoder in self.vit_multi[1:]:
                enout=encoder(enout[5])
                self.qxp5.append(enout[5])
                self.qxp6.append(enout[6])
        # self.encoder_outs = self.vit(transformer_input)
        en_out=self.qxp5[-1]
        input_mlp_head=torch.flatten(en_out[:,0:1,:],start_dim=1)
        mlp69_out=self.linear(input_mlp_head)
        return mlp69_out
    

class VIT_trainer():
    def __init__(self,model,epochs,optimizer,loss,trainingdata,validation_data,device=device):
        self.model = model.to(device)
        self.epochs = epochs
        self.optimizer = optimizer
        self.loss = loss
        self.trainingdata=trainingdata
        self.validation_data=validation_data
        self.device=device


    def train(self):
        self.train_loss=[]
        self.test_loss=[]
        self.train_accuracy=[]
        self.test_accuracy=[]
        for j in range(self.epochs):
            # training
            self.model.train()
            self.optimizer.zero_grad()
            losses=[]
            acc=[]
            for batch_idx, (images, labels) in enumerate(self.trainingdata):
                images=images.to(self.device)
                labels=labels.to(self.device)
                output=self.model(images)
                loss_=self.loss(output,labels)
                loss_.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                losses.append(loss_.item())

                predictions=torch.softmax(output,dim=-1)
                predictions=predictions.detach().numpy()
                pred_class=np.argmax(predictions,axis=-1)
                acc.append(accuracy(pred_class,labels))

            self.train_loss.append(np.mean(losses))
            self.train_accuracy.append(np.mean(acc))
            print(np.mean(losses))

            # validation
            self.model.eval()

            losses_=[]
            acc_=[]
            for batch_idx_, (images_, labels_) in enumerate(self.validation_data):
                images_=images_.to(self.device)
                labels_=labels_.to(self.device)
                out_test = self.model(images_)
                loss__=self.loss(out_test,labels_)
                losses_.append(loss__.item())

                predictions_=torch.softmax(out_test,dim=-1)
                predictions_=predictions_.detach().numpy()
                pred_class_=np.argmax(predictions_,axis=-1)
                acc_.append(accuracy(pred_class_,labels_))
            self.test_loss.append(np.mean(losses_))
            self.test_accuracy.append(np.mean(acc_))

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


def accuracy(preds,actual):
    count=0
    for i,j in zip(preds,actual):
        if i==j:
            count=count+1
    return count/len(actual)


## QUESTION 1

# if __name__ == "__main__":
set_seed(42)
"""Trainig VIT on cifar"""

channels, height, width = 3, 32, 32
patch_size = 4
emb_dim = 64 ## dimesion of embedding vector for each patches
stride = 4
num_class = 10 ## no of classes the model can classify an input
hidden = 128 ## hidden layer dimension of MLP head in the encoder
batch_size = 128
epochs = 30
num_att_head = 4 ## no of attention heads
train_perc = 0.9 ## split data into train and val
train_perc__ = 1 ## traing the model on lesser training data


## define model with required parameters
model = VIT_classification(num_class,channels,patch_size,emb_dim,hidden,height,width,stride,num_att_head)
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

## get trainloader and test loaders
data = get_data_loaders(batch_size,train_perc,train_perc__)

## inititae the trainer class which trains the model

trainer=VIT_trainer(model,epochs,optimizer,loss,data.trainloader,data.validationloader)
print("training has started")
trainer.train()
print("training has completed")

test_images = torch.stack([data[0] for data in data.testset])
test_labels = torch.tensor([data[1] for data in data.testset])

out_test = model(test_images)
predictions=torch.softmax(out_test,dim=-1)
predictions=predictions.detach().numpy()
pred_class=np.argmax(predictions,axis=-1)
print("The Test accuracy is",accuracy(pred_class,test_labels))

# plotting
plt.plot(range(epochs),trainer.train_loss,label="training loss")
plt.plot(range(epochs),trainer.test_loss,label="test loss")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend()
plt.savefig("plots/question1/loss.png")
# plt.show()
plt.close()

plt.plot(range(epochs),trainer.train_accuracy,label="training accuracy")
plt.plot(range(epochs),trainer.test_accuracy,label="test accuracy")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.legend()
plt.savefig("plots/question1/acc.png")
# plt.show()
plt.close()




## QUESTION 2


train_perc__=[0.05,0.1,0.25,0.5,1]
accuracies_q2=[]
losses_q2=[]
for i in train_perc__:
    print("The model will start training for ",i*100,"% of the training data.")
    model = VIT_classification(num_class,channels,patch_size,emb_dim,hidden,height,width,stride,num_att_head)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    data = get_data_loaders(batch_size,train_perc,i)

    ## inititae the trainer class which trains the model

    trainer=VIT_trainer(model,epochs,optimizer,loss,data.trainloader,data.validationloader)
    print("training has started")
    trainer.train()
    print("training has completed")

    test_images = torch.stack([data[0] for data in data.testset])
    test_labels = torch.tensor([data[1] for data in data.testset])

    out_test = model(test_images)
    losses_q2.append(loss(out_test,test_labels).item())
    predictions=torch.softmax(out_test,dim=-1)
    predictions=predictions.detach().numpy()
    pred_class=np.argmax(predictions,axis=-1)
    accuracies_q2.append(accuracy(pred_class,test_labels))
    print("The Test accuracy when the model is trained on ",i*100,"% of the training data is = ", accuracy(pred_class,test_labels))

# plotting
plt.plot(train_perc__,accuracies_q2,'go--',label="test accuracy")
plt.xlabel("Different sizes of training data on which the model is trained")
plt.ylabel("test accuracy")
plt.legend()
plt.savefig("plots/question2/acc.png")
# plt.show()
plt.close()

plt.plot(train_perc__,losses_q2,'go--',label="test loss")
plt.xlabel("Different sizes of training data on which the model is trained")
plt.ylabel("test loss")
plt.legend()
plt.savefig("plots/question2/loss.png")
# plt.show()
plt.close()


## QUESTION 3

patch_size_list=[4,8,16]

accuracies_q3=[]
losses_q3=[]

accuracies_q3_=[]
losses_q3_=[]

for i in patch_size_list:
## this part of code is for non - overlapping patches    
    print("The model will start training for patch size = ",i," and non - overlapping patches are used")
    model = VIT_classification(num_class,channels,i,emb_dim,hidden,height,width,i,num_att_head)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    data = get_data_loaders(batch_size)

    ## inititae the trainer class which trains the model

    trainer=VIT_trainer(model,epochs,optimizer,loss,data.trainloader,data.validationloader)
    print("training has started")
    trainer.train()
    print("training has completed")

    test_images = torch.stack([data[0] for data in data.testset])
    test_labels = torch.tensor([data[1] for data in data.testset])

    out_test = model(test_images)
    losses_q3.append(loss(out_test,test_labels).item())
    predictions=torch.softmax(out_test,dim=-1)
    predictions=predictions.detach().numpy()
    pred_class=np.argmax(predictions,axis=-1)
    accuracies_q3.append(accuracy(pred_class,test_labels))
    print("The Test accuracy when the model is trained on patch size = ",i," and non - overlapping patches are used = ", accuracy(pred_class,test_labels))

## this part of code is for overlapping patches

    print("The model will start training for patch size = ",i," and overlapping patches are used")
    default_stride=int(0.5*i)
    model_ = VIT_classification(num_class,channels,i,emb_dim,hidden,height,width,default_stride,num_att_head)
    loss_ = nn.CrossEntropyLoss()
    optimizer_ = torch.optim.Adam(model_.parameters(), lr=1e-3, weight_decay=1e-5)

    ## inititae the trainer class which trains the model

    trainer_ = VIT_trainer(model_,epochs,optimizer_,loss_,data.trainloader,data.validationloader)
    print("training has started")
    trainer_.train()
    print("training has completed")

    test_images = torch.stack([data[0] for data in data.testset])
    test_labels = torch.tensor([data[1] for data in data.testset])

    out_test_ = model_(test_images)
    losses_q3_.append(loss_(out_test_,test_labels).item())
    predictions_ = torch.softmax(out_test_,dim=-1)
    predictions_ = predictions_.detach().numpy()
    pred_class_=np.argmax(predictions_,axis=-1)
    accuracies_q3_.append(accuracy(pred_class_,test_labels))
    print("The Test accuracy when the model is trained on patch size = ",i," and overlapping patches are used = ", accuracy(pred_class_,test_labels))

# plotting
plt.plot(patch_size_list,accuracies_q3,label="test accuracy for non overlapping patches")
plt.plot(patch_size_list,accuracies_q3_,label="test accuracy for overlapping patches")
plt.xlabel("Different sizes of patch size on which the model is trained")
plt.ylabel("test accuracy")
plt.legend()
plt.savefig("plots/question3/acc.png")
# plt.show()
plt.close()

plt.plot(patch_size_list,losses_q3,label="test loss for non overlapping patches")
plt.plot(patch_size_list,losses_q3_,label="test loss for overlapping patches")
plt.xlabel("Different sizes of patch size on which the model is trained")
plt.ylabel("test loss")
plt.legend()
plt.savefig("plots/question3/loss.png")
# plt.show()
plt.close()




## QUESTION 4
attention_head = [1,4,8,16,64]
epochs=10
accuracies_q4=[]
losses_q4=[]
for i in attention_head:
    print("The model will start training for ",i," attention head as the model parameter")
    model = VIT_classification(num_class,channels,patch_size,emb_dim,hidden,height,width,stride,i)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    data = get_data_loaders(batch_size)

    ## inititae the trainer class which trains the model

    trainer=VIT_trainer(model,epochs,optimizer,loss,data.trainloader,data.validationloader)
    print("training has started")
    trainer.train()
    print("training has completed")

    test_images = torch.stack([data[0] for data in data.testset])[:2000].to(device)
    test_labels = torch.tensor([data[1] for data in data.testset])[:2000].to(device)
    out_test = model(test_images)
    losses_q4.append(loss(out_test,test_labels).item())
    predictions=torch.softmax(out_test,dim=-1)
    predictions=predictions.detach().numpy()
    pred_class=np.argmax(predictions,axis=-1)
    accuracies_q4.append(accuracy(pred_class,test_labels))
    print("The Test accuracy when the model is trained with ",i," attention heads is = ", accuracy(pred_class,test_labels))

# plotting
plt.plot(attention_head,accuracies_q4,'go--',label="test accuracy for model with different attention heads")
plt.xlabel("Different sizes of attention head on which the model is trained")
plt.ylabel("test accuracy")
plt.legend()
plt.savefig("plots/question4/acc.png")
# plt.show()
plt.close()

plt.plot(attention_head,losses_q4,'go--',label="test loss for model with different attention heads")
plt.xlabel("Different sizes of attention head on which the model is trained")
plt.ylabel("test loss")
plt.legend()
plt.savefig("plots/question4/loss.png")
# plt.show()
plt.close()



# QUESION 5
epochs = 10
## define model with required parameters
model = VIT_classification(num_class,channels,patch_size,emb_dim,hidden,height,width,stride,num_att_head,5)
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

## get trainloader and test loaders
data = get_data_loaders(batch_size)

## inititae the trainer class which trains the model

trainer=VIT_trainer(model,epochs,optimizer,loss,data.trainloader,data.validationloader)
print("training has started")
trainer.train()
print("training has completed")

test_images = torch.stack([data[0] for data in data.testset])
test_labels = torch.tensor([data[1] for data in data.testset])

model.forward(test_images)
accuracies=[]
losses=[]

for i in model.qxp5:
    encoder_l1 = torch.flatten(i[:,0:1,:],start_dim=1)
    preds_l1 = model.linear(encoder_l1)

    predictions_l1=torch.softmax(preds_l1,dim=-1)
    predictions_l1=predictions_l1.detach().numpy()
    pred_class_l1=np.argmax(predictions_l1,axis=-1)
    losses.append(loss(preds_l1,test_labels).item())
    accuracies.append(accuracy(pred_class_l1,test_labels))

str_="accuracy_plot.png"
plt.plot(range(model.num_encoder),accuracies,"ro--")
plt.xlabel("ith layer")
plt.ylabel("Test Accuracy")
plt.savefig("plots/question5/"+str_)
# plt.show()
plt.close()

str_="loss_plot.png"
plt.plot(range(model.num_encoder),losses,"ro--")
plt.xlabel("ith layer")
plt.ylabel("Test loss")
plt.savefig("plots/question5/"+str_)
# plt.show()
plt.close()

## QUESTION 6

epochs=15
## define model with required parameters
model = VIT_classification(num_class,channels,patch_size,emb_dim,hidden,height,width,stride,num_att_head,5)
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

## get trainloader and test loaders
data = get_data_loaders(batch_size)

## inititae the trainer class which trains the model

trainer=VIT_trainer(model,epochs,optimizer,loss,data.trainloader,data.validationloader)
print("training has started")
trainer.train()
print("training has completed")

test_images = torch.stack([data[0] for data in data.testset])
test_labels = torch.tensor([data[1] for data in data.testset])

num_classes = 10
samples_per_class = 2
selected_images = []
selected_labels = []

# Dictionary to track selected samples per class
class_counts = {i: 0 for i in range(num_classes)}

# Iterate over dataset and collect 2 samples per class
index=[]
idx=0
for img, label in zip(test_images, test_labels):
    if class_counts[label.item()] < samples_per_class:
        selected_images.append(img)
        selected_labels.append(label)
        class_counts[label.item()] += 1
        index.append(idx)
    idx=idx+1
    
    # Stop if we have enough samples
    if all(count == samples_per_class for count in class_counts.values()):
        break



## setting up data structures for future computation
obj_test=[]
test=[]
label=[]

for i,m in enumerate(index):
    obj_test.append(data.testset_object[m][0])
    test.append(data.testset[m][0])
    label.append(selected_labels[i].item())


obj_test_mapping = {x : [] for x in range(10)}
test_mapping = {x : [] for x in range(10)}
# label_mapping = {x : [] for x in range(10)}

for ot, t, l in zip(obj_test, test, label):
    obj_test_mapping[l].append(ot)
    test_mapping[l].append(t)


fig, ax = plt.subplots(10, 4, figsize=(6, 18))

class_names = ["airplane", "automobile", "bird", "cat", "deer", 
               "dog", "frog", "horse", "ship", "truck"]

for label in range(10):
    obj_tests = obj_test_mapping[label]
    tests = test_mapping[label]
    row_num = label
    for idx in range(2):
        pil_img = obj_tests[idx]
        X = torch.unsqueeze(tests[idx], 0)

        with torch.no_grad():
            scores = model(X.float().to(device))
            all_attentions = model.qxp6

        # print(scores.squeeze().cpu().numpy().shape)
        pred = np.argmax(scores.squeeze().cpu().numpy())
        # layers, batch_size, attention_heads, seq, seq
        total_attention = torch.stack(all_attentions)
        total_attention = total_attention.cpu()
        # layers, attention_heads, seq, seq because batch_size is 1
        total_attention = torch.squeeze(total_attention)
        # average the attention weights
        att_matix = torch.mean(total_attention, dim=1)

        residual_att = torch.eye(att_matix.size(1))
        concatenated_aatribute_matrix = att_matix + residual_att
        concatenated_aatribute_matrix = concatenated_aatribute_matrix / concatenated_aatribute_matrix.sum(dim=-1).unsqueeze(-1)

        # Recursively multiply the weight matrices
        joint_attentions = torch.zeros(concatenated_aatribute_matrix.size())
        joint_attentions[0] = concatenated_aatribute_matrix[0]

        for n in range(1, concatenated_aatribute_matrix.size(0)):
            joint_attentions[n] = torch.matmul(concatenated_aatribute_matrix[n], joint_attentions[n-1])

        for n in range(1, concatenated_aatribute_matrix.size(0)):
            joint_attentions[n] = torch.matmul(concatenated_aatribute_matrix[n], joint_attentions[n-1])

        v = joint_attentions[-1]
        grid_size = int(np.sqrt(concatenated_aatribute_matrix.size(-1)))
        mask = v[0, 1:].reshape(grid_size, grid_size).detach().numpy()
        result = cv2.resize(mask / mask.max(), pil_img.size)

        ax[row_num][2 * idx].imshow(pil_img)
        title = f"l = {class_names[label]}"
        ax[row_num][2 * idx].set_title(title)
        ax[row_num][2 * idx + 1].imshow(result,cmap="jet")
        title = f"p = {class_names[pred]}"
        ax[row_num][2 * idx + 1].set_title(title)


plt.savefig("maya.png")
plt.show()
