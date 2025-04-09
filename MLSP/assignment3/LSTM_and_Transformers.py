import torch
import random
from gensim.models import Word2Vec, FastText
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
import gensim.downloader as api
from gensim.models import KeyedVectors
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import re
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import librosa
# nltk.download("stopwords")
stop_words = set(stopwords.words("english"))
# nltk.download('punkt')  # Download tokenizer
# nltk.download('punkt_tab')
 
# loading and creating required dataset for model training



device = "cuda:3" if torch.cuda.is_available() else "cpu"
 
df = pd.read_csv('IMDB Dataset.csv')
 
w2v_model = api.load("word2vec-google-news-300")


#________________________________________________________________QUESTION 1A________________________________________________________________


train_index = random.sample(range(50000),40000)
test_index = list(set(range(50000))-set(train_index))


train_df = df.loc[train_index,:]


val_len = int(0.5*len(test_index))

val_df = df.loc[test_index[:val_len],:]

test_df = df.loc[test_index[val_len:],:]

def convert(train_df):
    result_train = list(set(re.sub(r'[^\w\s]', '', " ".join(train_df['review']).lower()).split()))
    dict_train ={"words":[],"embeddings":[]}
    for i in result_train:
        if i in w2v_model:
            dict_train["words"].append(i)
            dict_train["embeddings"].append(w2v_model[i])

    data_train = pd.DataFrame(dict_train)
    # Convert 'features' column into separate columns
    expanded_df_train = data_train.join(pd.DataFrame(data_train['embeddings'].tolist(), index=data_train.index))

    # Drop the original 'features' column if no longer needed
    expanded_df_train = expanded_df_train.drop(columns=['embeddings'])
    mat = expanded_df_train.loc[:,range(100)].values
    tensor = torch.tensor(mat)
    return tensor, expanded_df_train

tensor_train = convert(train_df)[0]
tensor_val = convert(val_df)[0]
tensor_test = convert(test_df)[0]

## PCA code
def PCA(train_data,test_data,num_features):

    """In this PCA code from scratch the top eigen vectors are calculated from traing data and 
        the transformation is done on any data given as argument under test data
    Attributes:
                train_data: eigen vectors are calculated using this data
                test_data: this data is projected to lower dimensions
                num_features: The dimension of the subspace on which data is being projeted 
    Returns:
                compreesed data with lower dimension"""
    
    

    train_mean=np.expand_dims(train_data.mean(axis=0), axis=1)
    normalized_data=train_data-train_mean.T
    covariance_mat=(normalized_data.T@normalized_data)/(train_data.shape[0])
    # covariance_mat=np.cov(normalized_data)
    eig_val,eig_vec=np.linalg.eigh(covariance_mat)


    #sort the eigenvalues and eigen vectors in descending order
    idx = eig_val.argsort()[::-1]   
    eigenValues = eig_val[idx]
    
    eigenVectors = eig_vec[:,idx]
    compressed_data=(test_data-train_mean.T)@eigenVectors[:,:num_features]
    return compressed_data

# import pdb;pdb.set_trace()
tensor_val_10, df = convert(val_df.head(10))

compressed_ = PCA(tensor_train,tensor_val_10,2)


first_column = df.iloc[:, 0]
tensor_df = pd.DataFrame(compressed_.numpy(), columns=["Value1", "Value2"])
new_df = pd.concat([first_column, tensor_df], axis=1)

vis_data = new_df
# .loc[:100,:]
plt.figure(figsize=(20, 20))
plt.scatter(vis_data["Value1"], vis_data["Value2"], color="blue")

# Add labels for each point using FirstCol
for i, txt in enumerate(vis_data["words"]):

    plt.annotate(txt, (vis_data["Value1"][i], vis_data["Value2"][i]), fontsize=8, ha='right')


# Labels and Title
plt.xlabel("Value1")
plt.ylabel("Value2")
plt.title("Scatter Plot with Labels")
plt.grid(True)

# Show the plot
plt.savefig("fig.png")
plt.show()


#________________________________________________________________QUESTION 1B________________________________________________________________

 
def sentence_to_matrix(sentence, model, emb_dim=300):
    """
    Convert a sentence into a matrix of shape (seq_len, emb_dim).
    """
    # Tokenize sentence
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(sentence)
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    # tokens = word_tokenize(sentence)
    # Get word embeddings, filter out OOV words
    embeddings = [model[word] for word in filtered_tokens if word in model]
    if not embeddings:
        return np.zeros((1, emb_dim), dtype=np.float32)  # Return a zero vector if no valid words
    return np.array(embeddings, dtype=np.float32)
 
 
 
class CustomDataset(Dataset):
    def __init__(self, data,indices):
        self.data = data
        self.indices = indices
 
    def __getitem__(self, idx):
        return (self.data["review"][self.indices[idx]],self.data["sentiment"][self.indices[idx]])
    
    def __len__(self):
        return len(self.indices)

    def collate_fn_transformer(self, batch):
        len_batch = len(batch)
        tensors = []
        labels = []
        for i in range(len_batch):
            mat = sentence_to_matrix(batch[i][0],w2v_model)
            tensors.append(torch.tensor(mat))
            if batch[i][1] =="positive":
                labels.append(1)
            else:
                labels.append(0)
        # max_rows = max(tensor.shape[0] for tensor in tensors)
        max_rows = 200
        padded_tensors = []
        for tensor in tensors:
            rows, cols = tensor.shape  # Get current shape
            if rows<max_rows:
                pad_size = (0, 0, 0, max_rows - rows)  # (pad cols, pad rows)
                padded_tensor = torch.nn.functional.pad(tensor, pad_size, "constant", 0)
                padded_tensors.append(padded_tensor)
            else:
                padded_tensors.append(tensor[:max_rows,:])
 
        final_tensors = torch.stack(padded_tensors, dim=0)
        return final_tensors, torch.tensor(labels)
 
 
    def collate_fn(self, batch):
        len_batch = len(batch)
        tensors = []
        labels = []
        for i in range(len_batch):
            mat = sentence_to_matrix(batch[i][0],w2v_model)
            tensors.append(torch.tensor(mat))
            if batch[i][1] =="positive":
                labels.append(1)
            else:
                labels.append(0)
        max_rows = max(tensor.shape[0] for tensor in tensors)
        padded_tensors = []
        for tensor in tensors:
            rows, cols = tensor.shape  # Get current shape
            pad_size = (0, 0, 0, max_rows - rows)  # (pad cols, pad rows)
            padded_tensor = torch.nn.functional.pad(tensor, pad_size, "constant", 0)
            padded_tensors.append(padded_tensor)
 
        final_tensors = torch.stack(padded_tensors, dim=0)
        return final_tensors, torch.tensor(labels)
 
 
 
 
class SentiModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = torch.nn.LSTM(300,256,2,batch_first=True)
        self.linear = torch.nn.Linear(256,2)
        # self.sig = torch.nn.Sigmoid()
        
    
    def forward(self,input):
        lstm_out , h_c = self.lstm(input)
        average_out = torch.mean(lstm_out,dim=1)
        linear_out = self.linear(average_out)
        # out = self.sig(linear_out)
        return linear_out
 
 
class SentiModel_att(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = torch.nn.LSTM(300,256,2,batch_first=True)
        self.linear = torch.nn.Linear(256,2)
        # self.sig = torch.nn.Sigmoid()
        self.att_vec = torch.nn.Parameter(torch.randn(256,1))
        
    
    def forward(self,input):
        lstm_out , h_c = self.lstm(input)
        # average_out = torch.mean(lstm_out,dim=1)
        matmul = torch.matmul(lstm_out,self.att_vec)
        matmul_t = torch.transpose(matmul,-2,-1)
        softy = torch.nn.functional.softmax(matmul_t,dim = -1)
        final = torch.matmul(softy,lstm_out)
        

        linear_out = self.linear(final)
        # out = self.sig(linear_out)
        return torch.squeeze(linear_out,dim = 1)


def accuracy(preds,actual):
    count=0
    for i,j in zip(preds,actual):
        if i==j:
            count=count+1
    return count/len(actual)
 
class Trainer:
    def __init__(self,model,epochs,trainloader,valloader,criterion,optimizer):
        self.epochs = epochs
        self.trainloader = trainloader
        self.valloader = valloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.model = model
 
        self.model.to(device)
 
    def train(self):
 
        self.loss_list = []
        self.loss_list_val = []
        self.acc_test = []
        for epoch in range(self.epochs):
            loss_epoch = []
            progress_bar = tqdm(self.trainloader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)
            self.optimizer.zero_grad()
            for idx,(x,y) in enumerate(progress_bar):
                x = x.to(device)
                y = y.to(device)
                out = self.model(x)
                loss = self.criterion(out,y)
                loss_epoch.append(loss.item())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                progress_bar.set_postfix(loss=loss.item())
            
            self.loss_list.append(np.mean(loss_epoch))
            print(f"Epoch {epoch+1} Loss: {self.loss_list[-1]:.4f}")
            
            loss_epoch_val = []
            for idx,(x_,y_) in enumerate(self.valloader):
                x_ = x_.to(device)
                y_ = y_.to(device)
                out_ = self.model(x_)
                loss_ = self.criterion(out_,y_)
                loss_epoch_val.append(loss_.item())
            self.loss_list_val.append(np.mean(loss_epoch_val))

            acc_ = []
            for x,y in testloader:
                x = x.to(device)
                y = y.to(device)
                out_test = self.model(x)
                predictions_=torch.softmax(out_test,dim=-1)
                predictions_=predictions_.detach().cpu().numpy()
                pred_class_=np.argmax(predictions_,axis=-1)
                acc_.append(accuracy(pred_class_,y))
            acc_test = np.mean(acc_)
            self.acc_test.append(acc_test)
            print("The test accuracy for the test data is ",acc_test)
 
 
epochs = 25
batch_size = 128
 
train_index = random.sample(range(50000),40000)
test_index = list(set(range(50000))-set(train_index))
 
trainset = CustomDataset(df, train_index)
 
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, collate_fn=trainset.collate_fn)
 

val_len = int(0.5*len(test_index))
val_index = test_index[:val_len]
test_index = test_index[val_len:]
 
valset = CustomDataset(df, val_index)

valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, collate_fn=valset.collate_fn)
 
testset = CustomDataset(df, test_index)
 
testloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, collate_fn=testset.collate_fn)
 
 
model =  SentiModel()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
 
trainer = Trainer(model,epochs,trainloader,valloader,criterion,optimizer)
 
trainer.train()
 
plt.plot(range(epochs),trainer.loss_list,'go--',label="train loss")
plt.plot(range(epochs),trainer.loss_list_val,'ro--',label="validation loss")
plt.legend()
plt.savefig("loss_curve_1.png")
plt.close()
 
plt.plot(range(epochs),trainer.acc_test,'go--',label="test accuracy")
plt.legend()
plt.savefig("test_acc_curve_1.png")
plt.close()
 
#________________________________________________________________QUESTION 1C________________________________________________________________

## attention pooling

model =  SentiModel_att()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
 
trainer = Trainer(model,epochs,trainloader,valloader,criterion,optimizer)
 
trainer.train()
plt.plot(range(epochs),trainer.loss_list,'go--',label="train loss")
plt.plot(range(epochs),trainer.loss_list_val,'ro--',label="validation loss")
plt.legend()
plt.savefig("loss_curve_2.png")
plt.close()
 
plt.plot(range(epochs),trainer.acc_test,'go--',label="test accuracy")
plt.legend()
plt.savefig("test_acc_curve_2.png")
plt.close()


#________________________________________________________________QUESTION 1D________________________________________________________________

 
## transformer model

class Embedding(torch.nn.Module):
    def __init__(self, input_dim ):
        super().__init__()
        self.clstoken=torch.nn.Parameter(torch.randn(1,1,input_dim))
        self.position=torch.nn.Embedding(200+1,input_dim)
        self.dropout = torch.nn.Dropout(0.2)

    def forward(self,input):
        """arguments:
                    image - input image is of size (batch_size, channels, height, width)
                    patch_size - size of patch we need from the image to be treated as a single sequence
                    input_dim - size of embedding vector
                    overlap - overlapping between patches of sequence
            returns:
                    embedding vector of each image concateneated with CLS token and positional embeddings.
                    output size - (batch_size, sequence_length, embedding dimension)"""
        batch_size = input.shape[0]
        embeddings=input
        cls=self.clstoken.expand(batch_size,-1,-1)
        embeddings_=torch.cat((cls,embeddings),dim=1)
        # position
        in_=torch.unsqueeze(torch.tensor(range(embeddings_.shape[1])),dim=0)
        in__=in_.expand(batch_size,-1).to(device)
        embs_=self.position(in__)
        added_positional_emb=embeddings_+embs_

        return added_positional_emb
 
class multimodal_self_attention_opt(torch.nn.Module):
    def __init__(self,num_attention_heads,input_dim):
        super().__init__()
        self.num_attention_heads=num_attention_heads
        self.input_dim=input_dim
        self.Q=torch.nn.Linear(input_dim,input_dim,bias=False)
        self.K=torch.nn.Linear(input_dim,input_dim,bias=False)
        self.V=torch.nn.Linear(input_dim,input_dim,bias=False)
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
        soft_dot_out = torch.nn.functional.softmax(dots_products,dim=-1)
        attentions = torch.matmul(soft_dot_out,values)
        
        attentions = attentions.permute(0, 2, 1, 3).contiguous()
        new_attention_layer_shape = attentions.size()[:-2] + (self.input_dim,)
        attentions = attentions.view(new_attention_layer_shape)
        return attentions

class MLP(torch.nn.Module):
    def __init__(self,input_dim,hid,out_dim):
        super().__init__()
        self.lin1=torch.nn.Linear(input_dim,hid)
        self.non_lin=torch.nn.ReLU()
        self.lin2=torch.nn.Linear(hid,out_dim)
 
    def forward(self,input):
        mlp_out_dummy = self.lin1(input)
        mlp_out_non=self.non_lin(mlp_out_dummy)
        mlp_out=self.lin2(mlp_out_non)
 
        return mlp_out
    
class LayerNorm(torch.nn.Module):
    def __init__(self,input_dim):
        super().__init__()
        self.layer_norm=torch.nn.LayerNorm(input_dim)
 
    def forward(self,embeddings):
        """arguments:
                    image - embeddings of shape (batch_size, sequence_length, embedding dimension)
            returns:
                    normalize each layer of embedding dimension.
                    output size - (batch_size, sequence_length, embedding dimension)"""
 
        normed_output=self.layer_norm(embeddings)
        return normed_output
    
class VIT_encoder(torch.nn.Module):
    def __init__(self,emb_dim,hid,num_att_head):
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
        add1=trans_in+att_out
        lm2=self.norm2(add1)
        mlp_=self.mlp1(lm2)
        add2=add1+mlp_
        return add2
 
class transformer_classification(torch.nn.Module):
    def __init__(self,num_classes,emb_dim,hid,num_att_head,num_encoder=1):
        super().__init__()
        self.num_encoder=num_encoder
        self.ve=Embedding(emb_dim)
        self.vit=VIT_encoder(emb_dim,hid,num_att_head)
        self.vit_multi=torch.nn.ModuleList([self.vit for i in range(self.num_encoder)])
        self.linear=torch.nn.Linear(emb_dim,num_classes)
    def forward(self,image):
        """arguments:
                    image - takes image (batch_size, chanel, height , width)
            returns:
                    The logits of no of classes to predict.
                    output size - (batch_size, number of classses)"""
        transformer_input=self.ve(image)
        ## below block of code returns the output of different layer of VIT encoder
        
        enout=self.vit_multi[0](transformer_input)
     
        if self.num_encoder>1:
            for encoder in self.vit_multi[1:]:
                enout=encoder(enout)
        input_mlp_head=torch.flatten(enout[:,0:1,:],start_dim=1)
        mlp69_out=self.linear(input_mlp_head)
        return mlp69_out


trainset = CustomDataset(df, train_index)
 
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, collate_fn=trainset.collate_fn_transformer)

valset = CustomDataset(df, val_index)
 
valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, collate_fn=valset.collate_fn_transformer)
 
testset = CustomDataset(df, test_index)
 
testloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, collate_fn=testset.collate_fn_transformer)


model_trans =  transformer_classification(2,300,70,1)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_trans.parameters(), lr=1e-3, weight_decay=1e-4)
 
trainer = Trainer(model_trans,epochs,trainloader,valloader,criterion,optimizer)
trainer.train()

plt.plot(range(epochs),trainer.loss_list,'go--',label="train loss")
plt.plot(range(epochs),trainer.loss_list_val,'ro--',label="validation loss")
plt.legend()
plt.savefig("loss_curve_3.png")
plt.close()
 
plt.plot(range(epochs),trainer.acc_test,'go--',label="test accuracy")
plt.legend()
plt.savefig("test_acc_curve_3.png")
plt.close()





#________________________________________________________________QUESTION 3________________________________________________________________



device = "cpu"
df = pd.read_csv('esc50.csv')
esc10_df = df.loc[df["esc10"] == True]

## sanity
esc10_df['target'] = esc10_df['target'].map({0:0, 1:1, 38:2, 40:3, 41:4, 10:5, 11:6, 12:7, 20:8, 21:9})

train_list_file_paths = esc10_df.loc[esc10_df["fold"].isin([1,2,3])]["filename"].to_list()
train_list_y = esc10_df.loc[esc10_df["fold"].isin([1,2,3])]["target"].to_list()

val_list_file_paths = esc10_df.loc[esc10_df["fold"].isin([4])]["filename"].to_list()
val_list_y = esc10_df.loc[esc10_df["fold"].isin([4])]["target"].to_list()

test_list_file_paths = esc10_df.loc[esc10_df["fold"].isin([5])]["filename"].to_list()
test_list_y = esc10_df.loc[esc10_df["fold"].isin([5])]["target"].to_list()
path = 'audio/'

def extract_mel_spectrogram(audio_path, n_mels=128, win_ms=25, hop_ms=10, duration=5, sr=44100):
    ap = path+audio_path
    y, sr = librosa.load(ap, sr=None, duration=duration)
    win_length = int(win_ms * sr / 1000)
    hop_length = int(hop_ms * sr / 1000)
    
    mel_spectrogram = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=n_mels,
        win_length=win_length,
        hop_length=hop_length,
    )
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    
    return mel_spectrogram_db

train_data = []
for i in train_list_file_paths:
    train_data.append(extract_mel_spectrogram(i))

val_data = []
for i in val_list_file_paths:
    val_data.append(extract_mel_spectrogram(i))

test_data = []
for i in test_list_file_paths:
    test_data.append(extract_mel_spectrogram(i))

train_x = torch.tensor(np.array(train_data))
val_x = torch.tensor(np.array(val_data))
test_x = torch.tensor(np.array(test_data))

train_y = torch.tensor(np.array(train_list_y))
val_y = torch.tensor(np.array(val_list_y))
test_y = torch.tensor(np.array(test_list_y))

class CustomDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x = x_data
        self.y = y_data

    def __len__(self):
        return len(self.x)

    def __getitem__(self,idx):
        x = self.x[idx,:,:]
        y = self.y[idx]
        return x, y
    

class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.con1 = torch.nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = 3, stride=1)
        self.act1 = torch.nn.ReLU()
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size = 3, stride=None)
        self.con2 = torch.nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = 3, stride=1)
        self.act2 = torch.nn.ReLU()
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size = 3, stride=None)
        self.linear1 = torch.nn.Linear(11232,128)
        self.act3 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(128,10)

    def forward(self,input):
        
        input = torch.unsqueeze(input,dim = 1)
        con1_out = self.con1(input)
        act1 = self.act1(con1_out)
        max1_out = self.maxpool1(act1)
        con2_out = self.con2(max1_out)
        act2 = self.act2(con2_out)
        max2_out = self.maxpool2(act2)
        flat = torch.flatten(max2_out,start_dim = 1)
        l1_out = self.linear1(flat)
        act3 = self.act3(l1_out)
        l2_out = self.linear2(act3)
        return l2_out






def accuracy(preds,actual):
    count=0
    for i,j in zip(preds,actual):
        if i==j:
            count=count+1
    return count/len(actual)
 
class Trainer:
    def __init__(self,model,epochs,trainloader,valloader,testloader,criterion,optimizer):
        self.epochs = epochs
        self.trainloader = trainloader
        self.valloader = valloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.model = model
        self.testloader = testloader
        self.model.to(device)
 
    def train(self):
 
        self.loss_list = []
        self.loss_list_val = []
        self.acc_test = []
        for epoch in range(self.epochs):
            loss_epoch = []
            # progress_bar = tqdm(self.trainloader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)
            self.optimizer.zero_grad()
            for idx,(x,y) in enumerate(trainloader):
                x = x.to(device)
                y = y.to(device)
                out = self.model(x)
                loss = self.criterion(out,y)
                loss_epoch.append(loss.item())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # progress_bar.set_postfix(loss=loss.item())
            
            self.loss_list.append(np.mean(loss_epoch))
            # print(f"Epoch {epoch+1} Loss: {self.loss_list[-1]:.4f}")
            
            loss_epoch_val = []
            for idx,(x_,y_) in enumerate(self.valloader):
                x_ = x_.to(device)
                y_ = y_.to(device)
                out_ = self.model(x_)
                loss_ = self.criterion(out_,y_)
                loss_epoch_val.append(loss_.item())
            self.loss_list_val.append(np.mean(loss_epoch_val))

            acc_ = []
            for x,y in self.testloader:
                x = x.to(device)
                y = y.to(device)
                out_test = self.model(x)
                predictions_=torch.softmax(out_test,dim=-1)
                predictions_=predictions_.detach().cpu().numpy()
                pred_class_=np.argmax(predictions_,axis=-1)
                acc_.append(accuracy(pred_class_,y))
            acc_test = np.mean(acc_)
            self.acc_test.append(acc_test)
            # print("The test accuracy for the test data is ",acc_test)
 




trainset = CustomDataset(train_x,train_y)
trainloader = torch.utils.data.DataLoader(trainset,batch_size = 16)

valset = CustomDataset(val_x,val_y)
valloader = torch.utils.data.DataLoader(valset,batch_size = 16)

testset = CustomDataset(test_x,test_y)
testloader = torch.utils.data.DataLoader(testset,batch_size = 16)

epochs =10


 

#________________________________________________________________QUESTION 3A________________________________________________________________

#SGD
model_sgd = CNN()

criterion_sgd = torch.nn.CrossEntropyLoss()
optimizer_sgd = torch.optim.SGD(model_sgd.parameters(),lr = 1e-4)
 
trainer_sgd = Trainer(model_sgd,10,trainloader,valloader,testloader,criterion_sgd,optimizer_sgd)
 
trainer_sgd.train()

plt.plot(range(epochs),trainer_sgd.loss_list,'go--',label="train loss")
plt.plot(range(epochs),trainer_sgd.loss_list_val,'ro--',label="validation loss")
plt.legend()
plt.savefig("plotsq3/loss_curve_sgd.png")
# plt.show()
plt.close()

# SGD with momentum

model_sgd_mom = CNN()

criterion_sgd_mom = torch.nn.CrossEntropyLoss()
optimizer_sgd_mom = torch.optim.SGD(model_sgd_mom.parameters(),lr = 1e-4, momentum=0.9)
 
trainer_sgd_mom = Trainer(model_sgd_mom,10,trainloader,valloader,testloader,criterion_sgd_mom,optimizer_sgd_mom)
 
trainer_sgd_mom.train()

plt.plot(range(epochs),trainer_sgd_mom.loss_list,'go--',label="train loss")
plt.plot(range(epochs),trainer_sgd_mom.loss_list_val,'ro--',label="validation loss")
plt.legend()
plt.savefig("plotsq3/loss_curve_sgdm.png")
# plt.show()
plt.close()

#RMS prop

model_rms = CNN()

criterion_rms = torch.nn.CrossEntropyLoss()
optimizer_srms = torch.optim.RMSprop(model_rms.parameters(),lr = 1e-4)
 
trainer_rms = Trainer(model_rms,10,trainloader,valloader,testloader,criterion_rms,optimizer_srms)
 
trainer_rms.train()

plt.plot(range(epochs),trainer_rms.loss_list,'go--',label="train loss")
plt.plot(range(epochs),trainer_rms.loss_list_val,'ro--',label="validation loss")
plt.legend()
plt.savefig("plotsq3/loss_curve_rms.png")
# plt.show()
plt.close()

#ADAM

model = CNN()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
 
trainer = Trainer(model,15,trainloader,valloader,testloader,criterion,optimizer)
 
trainer.train()
 
plt.plot(range(epochs),trainer.loss_list,'go--',label="train loss")
plt.plot(range(epochs),trainer.loss_list_val,'ro--',label="validation loss")
plt.legend()
plt.savefig("plotsq3/loss_curve_adam.png")
# plt.show()
plt.close()
 

plt.plot(range(epochs),trainer_sgd.loss_list,'go--',label="train loss sgd")
plt.plot(range(epochs),trainer_sgd_mom.loss_list,'bo--',label="train loss sgd with momentum")
plt.plot(range(epochs),trainer_rms.loss_list,'ro--',label="train loss rmsprop")
plt.plot(range(epochs),trainer.loss_list,'ko--',label="train loss adam")
plt.legend()
plt.xlabel("epochs")
plt.ylabel("training loss")
plt.savefig("plotsq3/loss_curve_comp_train.png")
plt.close()


plt.plot(range(epochs),trainer_sgd.loss_list_val,'go--',label="val loss sgd")
plt.plot(range(epochs),trainer_sgd_mom.loss_list_val,'bo--',label="val loss sgd with momentum")
plt.plot(range(epochs),trainer_rms.loss_list_val,'ro--',label="val loss rmsprop")
plt.plot(range(epochs),trainer.loss_list_val,'ko--',label="val loss adam")
plt.legend()
plt.xlabel("epochs")
plt.ylabel("training loss")
plt.savefig("plotsq3/loss_curve_comp_val.png")
plt.close()

print("The test accuracy for SGD is ",trainer_sgd.acc_test[-1])
print("The test accuracy for SGD with momentum is ",trainer_sgd_mom.acc_test[-1])
print("The test accuracy for RMS prop is ",trainer_rms.acc_test[-1])
print("The test accuracy for ADAM is ",trainer.acc_test[-1])



#________________________________________________________________QUESTION 3B________________________________________________________________

class CNN_b(torch.nn.Module):
    def __init__(self,norm=None):
        super().__init__()
        self.con1 = torch.nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = 3, stride=1)
        self.act1 = torch.nn.ReLU()
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size = 3, stride=None)
        self.con2 = torch.nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = 3, stride=1)
        self.act2 = torch.nn.ReLU()
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size = 3, stride=None)
        self.linear1 = torch.nn.Linear(11232,128)
        self.act3 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(128,10)
        self.norm_str = norm
        if self.norm_str == "batch":
            self.norm = torch.nn.BatchNorm1d(num_features= 11232)
        elif self.norm_str == "layer":
            self.norm = torch.nn.LayerNorm(normalized_shape = 11232)

    def forward(self,input):
        
        input = torch.unsqueeze(input,dim = 1)
        con1_out = self.con1(input)
        act1 = self.act1(con1_out)
        max1_out = self.maxpool1(act1)
        con2_out = self.con2(max1_out)
        act2 = self.act2(con2_out)
        max2_out = self.maxpool2(act2)
        flat = torch.flatten(max2_out,start_dim = 1)
        if self.norm_str is not None:
            flat = self.norm(flat)
        l1_out = self.linear1(flat)
        act3 = self.act3(l1_out)
        l2_out = self.linear2(act3)
        return l2_out

model_b = CNN_b()

criterion_b = torch.nn.CrossEntropyLoss()
optimizer_b = torch.optim.Adam(model_b.parameters(),lr = 1e-4)
 
trainer_b = Trainer(model_b,epochs,trainloader,valloader,testloader,criterion_b,optimizer_b)

trainer_b.train()

# layer norm
 

model_bl = CNN_b(norm="layer")

criterion_bl = torch.nn.CrossEntropyLoss()
optimizer_bl = torch.optim.Adam(model_bl.parameters(),lr = 1e-4)
 
trainer_bl = Trainer(model_bl,epochs,trainloader,valloader,testloader,criterion_bl,optimizer_bl)
 
trainer_bl.train()


# batch norm

model_bb = CNN_b(norm="batch")

criterion_bb = torch.nn.CrossEntropyLoss()
optimizer_bb = torch.optim.Adam(model_bb.parameters(),lr = 1e-4)
 
trainer_bb = Trainer(model_bb,epochs,trainloader,valloader,testloader,criterion_bb,optimizer_bb)
 
trainer_bb.train()

plt.plot(range(epochs),trainer_b.loss_list,'go--',label="train loss with no norm")
plt.plot(range(epochs),trainer_b.loss_list_val,'bo--',label="val loss with no norm")
plt.legend()
plt.xlabel("epochs")
plt.ylabel("loss")
plt.savefig("plotsq3/b/loss_curve_no.png")
plt.close()

plt.plot(range(epochs),trainer_bl.loss_list,'go--',label="train loss with layer norm")
plt.plot(range(epochs),trainer_bl.loss_list_val,'bo--',label="val loss with layer norm")
plt.legend()
plt.xlabel("epochs")
plt.ylabel("loss")
plt.savefig("plotsq3/b/loss_curve_l.png")
plt.close()

plt.plot(range(epochs),trainer_bb.loss_list,'go--',label="train loss with batch norm")
plt.plot(range(epochs),trainer_bb.loss_list_val,'bo--',label="val loss with batch norm")
plt.legend()
plt.xlabel("epochs")
plt.ylabel("loss")
plt.savefig("plotsq3/b/loss_curve_b.png")
plt.close()


plt.plot(range(epochs),trainer_b.loss_list,'go--',label="train loss with no norm")
plt.plot(range(epochs),trainer_bl.loss_list,'bo--',label="train loss with layer norm")
plt.plot(range(epochs),trainer_bb.loss_list,'ro--',label="train loss with batch norm")
plt.legend()
plt.xlabel("epochs")
plt.ylabel("training loss")
plt.savefig("plotsq3/b/loss_curve_comp_train.png")
plt.close()


plt.plot(range(epochs),trainer_b.loss_list_val,'go--',label="val loss with no norm")
plt.plot(range(epochs),trainer_bl.loss_list_val,'bo--',label="val loss with layer norm")
plt.plot(range(epochs),trainer_bb.loss_list_val,'ro--',label="val loss with batch norm")
plt.legend()
plt.xlabel("epochs")
plt.ylabel("training loss")
plt.savefig("plotsq3/b/loss_curve_comp_val.png")
plt.close()




#________________________________________________________________QUESTION 3C________________________________________________________________

# first model
model_sgd = CNN()

criterion_sgd = torch.nn.CrossEntropyLoss()
optimizer_sgd = torch.optim.SGD(model_sgd.parameters(),lr = 1e-4)
 
trainer_sgd = Trainer(model_sgd,epochs,trainloader,valloader,testloader,criterion_sgd,optimizer_sgd)
 
trainer_sgd.train()

# Second Model
class CNN_batchnorm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = torch.nn.Sequential(torch.nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = 3, stride=1),
                                        torch.nn.LayerNorm([16,126,499]),
                                        torch.nn.ReLU(),
                                        torch.nn.MaxPool2d(kernel_size = 3, stride=None),
                                        torch.nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = 3, stride=1),
                                        torch.nn.LayerNorm([16,40,164]),
                                        torch.nn.ReLU(),
                                        torch.nn.MaxPool2d(kernel_size = 3, stride=None))

        self.flat_seq = torch.nn.Sequential(torch.nn.Linear(11232,128),
                                            torch.nn.LayerNorm([128]),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(128,10))
        

    def forward(self,input):
        
        input = torch.unsqueeze(input,dim = 1)
        block1 = self.seq(input)
        flat = torch.flatten(block1,start_dim=1)
        out = self.flat_seq(flat)
        return out
    

model_lay = CNN_batchnorm()
criterion_lay = torch.nn.CrossEntropyLoss()
optimizer_lay = torch.optim.Adam(model_lay.parameters(), lr=1e-4)
 
trainer_lay = Trainer(model_lay,epochs,trainloader,valloader,testloader,criterion_lay,optimizer_lay)
 
trainer_lay.train()

## third model
class CNN_batchnorm(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.seq = torch.nn.Sequential(torch.nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = 3, stride=1),
                                        torch.nn.BatchNorm2d(16),
                                        torch.nn.ReLU(),
                                        torch.nn.MaxPool2d(kernel_size = 3, stride=None),
                                        torch.nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = 3, stride=1),
                                        torch.nn.BatchNorm2d(16),
                                        torch.nn.ReLU(),
                                        torch.nn.MaxPool2d(kernel_size = 3, stride=None))

        self.flat_seq = torch.nn.Sequential(torch.nn.Linear(11232,128),
                                            torch.nn.BatchNorm1d(128),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(128,10))
        

    def forward(self,input):
        
        input = torch.unsqueeze(input,dim = 1)
        block1 = self.seq(input)
        flat = torch.flatten(block1,start_dim=1)
        out = self.flat_seq(flat)
        return out
    

model = CNN_batchnorm()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
 
trainer = Trainer(model,epochs,trainloader,valloader,testloader,criterion,optimizer)
 
trainer.train()



# part A

acc_1 = []
for x,y in testloader:
    x = x.to(device)
    y = y.to(device)
    out_test = model_sgd(x)
    out_test_layer = model_lay(x)
    out_test_batch = model(x)
    

    ensemble_logits = out_test + out_test_layer + out_test_batch
    ensemble_logits = ensemble_logits/3

    predictions_=torch.softmax(ensemble_logits,dim=-1)
    

    predictions_=predictions_.detach().cpu().numpy()
    
    pred_class_=np.argmax(predictions_,axis=-1)
    acc_1.append(accuracy(pred_class_,y))


    


print("The test accuracy for the test data of ensemble model with simple average is ",np.mean(acc_1))


# Part B

alphas = torch.nn.Parameter(torch.rand(1,3))
cri = torch.nn.CrossEntropyLoss()
opt = torch.optim.Adam([alphas])

acc_epochs = []
for _ in range(20):
    opt.zero_grad()
    acc__ = []
    for x,y in testloader:
        x = x.to(device)
        y = y.to(device)
        out_test = model_sgd(x)
        out_test_layer = model_lay(x)
        out_test_batch = model(x)
        ensembele = (alphas[0][0]*out_test) + (alphas[0][1]*out_test_layer) + (alphas[0][2]*out_test_batch)
        opt.zero_grad()
        loss = cri(ensembele,y)
        loss.backward()
        opt.step()
        
        predictions_=torch.softmax(ensembele,dim=-1)

        predictions_=predictions_.detach().cpu().numpy()
        pred_class_=np.argmax(predictions_,axis=-1)
        acc__.append(accuracy(pred_class_,y))
    acc_epochs.append(np.mean(acc__))

print("The test accuracy for the test data of ensemble model with weighted average (optimal weights) is ",acc_epochs[-1])

