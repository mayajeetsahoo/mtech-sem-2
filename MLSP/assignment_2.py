import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
seed = 26
np.random.seed(seed)
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import random
from sklearn.svm import SVC
import seaborn as sns
random.seed(46)

"""__________________________________________________________QUESTION 2__________________________________________________________"""


# Loading data and converting images into numpy array for further implementation of algorithms

train_dir_path = "Data/emotion_classification/train"

X_train=[]
Y_train=[]
for file in os.listdir(train_dir_path):
    file_path=os.path.join(train_dir_path,file)
    if file_path[-5:-4]=="d":
        Y_train.append(0)
    else:
        Y_train.append(1)
    im=np.array(Image.open(file_path))
    X_train.append(im.ravel())
X_train=np.array(X_train)/255
Y_train=np.expand_dims(np.array(Y_train),axis=1)

test_dir_path = "Data/emotion_classification/test"

X_test=[]
Y_test=[]
for file in os.listdir(test_dir_path):
    file_path=os.path.join(test_dir_path,file)
    if file_path[-5:-4]=="d":
        Y_test.append(0)
    else:
        Y_test.append(1)
    im=np.array(Image.open(file_path))
    X_test.append(im.ravel())
X_test=np.array(X_test)/255
Y_test=np.expand_dims(np.array(Y_test),axis=1)




def PCA_high_dim(train_data,test_data,num_features):

    """In this PCA code from scratch the top eigen vectors are calculated from traing data and 
        the transformation is done on any data given as argument under test data
    Attributes:
                train_data: eigen vectors are calculated using this data
                test_data: this data is projected to lower dimensions
                num_features: The dimension of the subspace on which data is being projeted 
    Returns:
                compreesed data with lower dimension"""
    

    if num_features>19:
        raise ValueError("compression features should be less than N-1 for high dimensional PCA i.e. 19")
    train_mean=np.expand_dims(train_data.mean(axis=0), axis=1)
    normalized_data=train_data-train_mean.T
    # covariance_mat=(normalized_data@normalized_data.T)/(train_data.shape[0]-1)
    covariance_mat=np.cov(normalized_data)
    eig_val,eig_vec=np.linalg.eigh(covariance_mat)


    #sort the eigenvalues and eigen vectors in descending order
    idx = eig_val.argsort()[::-1]   
    eigenValues = eig_val[idx]
    
    eigenVectors = eig_vec[:,idx]
    original_eigen_vectors=(1/(np.sqrt(train_data.shape[0]*np.expand_dims(eigenValues[:-1],axis=0))))*((normalized_data.T@eigenVectors)[:,:-1])
    compressed_data=(test_data-train_mean.T)@original_eigen_vectors[:,:num_features]
    return compressed_data


def LDA_after_PCA(data,num_features):
    """In this LDA_after_PCA code from scratch the largest eigen vector is calculated form PCA compressed traing data by 
        applying the FISHER LINEAR DISCRIMINANT on the dimensionally reduced trainig data and then using it on the final data passed in argument.
    Attributes:
                data: Data on which PCA to num_features dimensions and LDA to one dimension is applied.
                num_features: the final dimension of PCA analysis
    Returns:
                compreesed data with one dimension"""
    
    compressed_train_data=PCA_high_dim(X_train,X_train,num_features)

    compressed_train_data_with_label=np.concatenate((compressed_train_data,np.array(Y_train)),axis=1)

    mean=np.expand_dims(compressed_train_data_with_label.mean(axis=0)[:-1],axis=0)
    mean0=np.expand_dims(compressed_train_data_with_label[np.where(compressed_train_data_with_label[:,-1].astype(int)==0)].mean(axis=0)[:-1],axis=0)
    mean1=np.expand_dims(compressed_train_data_with_label[np.where(compressed_train_data_with_label[:,-1].astype(int)==1)].mean(axis=0)[:-1],axis=0)


    S_b=(mean1.T-mean0.T)@((mean1.T-mean0.T).T)
    class_0_mat=compressed_train_data_with_label[np.where(compressed_train_data_with_label[:,-1].astype(int)==0)][:,:-1]-mean0
    class_0_cov=(class_0_mat.T@class_0_mat)/class_0_mat.shape[0]

    class_1_mat=compressed_train_data_with_label[np.where(compressed_train_data_with_label[:,-1].astype(int)==1)][:,:-1]-mean1
    class_1_cov=class_1_mat.T@class_1_mat/class_1_mat.shape[0]

    S_w=class_0_cov+class_1_cov

    # eigen_vector_for_1d_LDA=np.linalg.eigh(np.linalg.inv(S_w)@S_b)[1][:,-1]
    eigen_vector_for_1d_LDA=(np.linalg.inv(S_w)@(mean0-mean1).T)/np.linalg.norm((np.linalg.inv(S_w)@(mean0-mean1).T))

    compressed_data=PCA_high_dim(X_train,data,num_features)

    return compressed_data@eigen_vector_for_1d_LDA

def accuracy(y_pred,y_act):
    count=0
    for i in range(len(y_pred)):
        if y_pred[i]==y_act[i]:
            count=count+1

    return count/len(y_pred)

X_train_comp=PCA_high_dim(X_train,X_train,10)
X_test_comp=PCA_high_dim(X_train,X_test,10)

kernels = ["linear", "poly", "rbf", "sigmoid"]
k_acc=[]
for i in kernels:
    svmc = SVC(C=1.0, kernel=i, tol=1e-3)
    svmc.fit(X_train_comp, np.squeeze(Y_train))
    y_test_preds = svmc.predict(X_test_comp)
    acc=accuracy(y_test_preds,np.squeeze(Y_train))
    k_acc.append(acc)
sns.barplot(x=kernels, y=k_acc)
plt.xlabel("Kernel choice")
plt.ylabel("Test accuracy")
plt.title("SVM Test accuracy vs various choice of kernels")
plt.show()


Cs = [0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
Cs_acc=[]
for i in Cs:
    svmc = SVC(C=i, kernel="linear", tol=1e-3)
    svmc.fit(X_train_comp, np.squeeze(Y_train))
    y_test_preds = svmc.predict(X_test_comp)
    acc=accuracy(y_test_preds,np.squeeze(Y_train))
    Cs_acc.append(acc)
plt.plot(Cs,Cs_acc)
plt.xlabel("C")
plt.ylabel("Test accuracy")
plt.show()

tols = [1e-3, 1e-2, 1e-1, 1, 10]
tols_acc=[]
for i in tols:
    svmc = SVC(C=1.0, kernel="linear", tol=i)
    svmc.fit(X_train_comp, np.squeeze(Y_train))
    y_test_preds = svmc.predict(X_test_comp)
    acc=accuracy(y_test_preds,np.squeeze(Y_train))
    tols_acc.append(acc)
plt.plot(tols,tols_acc)
plt.xlabel("tolerance choice")
plt.xscale("log")
plt.ylabel("Test accuracy")
plt.show()



for j in kernels:
    K_acc=[]
    for i in range(4,18,2):
        X_train_comp=PCA_high_dim(X_train,X_train,i)
        X_test_comp=PCA_high_dim(X_train,X_test,i)
        svmc = SVC(C=i, kernel=j, tol=1e-3)
        svmc.fit(X_train_comp, np.squeeze(Y_train))
        y_test_preds = svmc.predict(X_test_comp)
        acc=accuracy(y_test_preds,np.squeeze(Y_train))
        K_acc.append(acc)
    plt.plot(range(4,18,2),K_acc,label=j)
plt.xlabel("K parameter of PCA")
plt.ylabel("Test accuracy")
plt.legend()
plt.show()



"""compression of traning and test data to ! dimension"""
reduced_train=LDA_after_PCA(X_train,10)
reduced_test=LDA_after_PCA(X_test,10)

"""Plotting one dimensional feature for each image for train and test data"""

colors=['red','blue']
for n in range(len(reduced_train)):
    plt.scatter(reduced_train[n], 0, color = colors[Y_train[n].item()])

colors=['black','green']
for n in range(len(reduced_test)):
    plt.scatter(reduced_test[n], 0.4, color = colors[Y_test[n].item()])

plt.xlabel("LDA dimension")
plt.ylabel("Y-axis(nothing significant)")
plt.show()

accuracy_hist=[]
threshold_hist=[]
for i in reduced_train:
    threshold_hist.append(i.item())
    target=np.where(reduced_train<=i,1,0)

    sum_=sum(a == b for a,b in zip(target, Y_train))
    accuracy=sum_/len(reduced_train)
    
    accuracy_hist.append(accuracy.item())


thresh=threshold_hist[np.argmax(accuracy_hist)]
threshold_hist.sort()
threshold=(thresh+threshold_hist[threshold_hist.index(thresh)+1])/2
print("The best accuracy  for training data is %s and is obtained at threshold %s"%(max(accuracy_hist)*100,threshold))

"""Using the above calculated threshold to separate test data points."""
target=np.where(reduced_test<threshold,1,0)
sum_=sum(a == b for a,b in zip(target, Y_test))
accuracy=sum_/len(reduced_test)

print("The accuracy for test data is %s"%(accuracy.item()*100))

"""__________________________________________________________QUESTION 3__________________________________________________________"""

corpus = pd.read_csv("Sentiment_Classification.txt", delimiter='\t', engine='python', names=["review", "label"])
reviews, labels = corpus.review.values.tolist(), corpus.label.values.tolist()
vectorizer = TfidfVectorizer(stop_words = 'english', min_df = 2)
tf_idf_matrix = vectorizer.fit_transform(reviews)

X = tf_idf_matrix.toarray()
Y=np.expand_dims(np.array(labels),1)
print(X.shape,Y.shape)

def train_val_split(x,y,perc):
   bar=int(np.round(perc*len(x)))
   sequence = random.sample(range(len(x)), len(x))
   return x[sequence[:bar],:],y[sequence[:bar],:],x[sequence[bar:],:],y[sequence[bar:],:]

X_train,y_train,X_test,y_test=train_val_split(X,Y,0.9)

def get_minibatch(training_x=X_train, training_y=y_train, batchSize=32):
    ## Read about Python generators if required.
    mini_x = []
    mini_y = []
    sequence = random.sample(range(len(training_x)), len(training_x))
    count = 1
    for i in range(len(training_x)):
      mini_x.append(training_x[sequence[i]])
      mini_y.append(training_y[sequence[i]])
      if(len(mini_x) == batchSize):
        yield np.array(mini_x), np.array(mini_y)
        mini_x = []
        mini_y = []
    if mini_x:
      yield np.array(mini_x), np.array(mini_y)


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

X_train_comp_=PCA(X_train,X_train,30)
X_train_comp,y_train_,X_val,y_val=train_val_split(X_train_comp_,y_train,0.9)

def trainer(epochs,learning_rate,batch_size,reg_param=0):
    # weight=np.random.normal(0,1,size=(X_train_comp.shape[1],1))
    weight=np.zeros((X_train_comp.shape[1],1))
    b=0
    loss_train=[]
    loss_val=[]
    for i in range(epochs):
        mini=get_minibatch(X_train_comp,y_train_,batchSize=batch_size)
        losses=[]
        for i,data in enumerate(mini):
            y_linear=(data[0]@weight)+b
            y_hat=1/(1+np.exp(-1*y_linear))
            loss_=-1*(data[1]*np.log(y_hat)+(1-data[1])*np.log(1-y_hat))
            loss=np.mean(loss_)
            prev_grad=(y_hat-data[1])/len(y_hat)
            grad_w=data[0].T@prev_grad
            grad_b=np.sum(prev_grad)
            weight=weight*(1-reg_param*learning_rate)-(learning_rate)*grad_w
            b=b-(learning_rate)*grad_b
            losses.append(loss)
        loss_train.append(np.mean(losses))
        mini=get_minibatch(X_val,y_val,batchSize=batch_size)
        losses=[]
        for j,data_ in enumerate(mini):
            
            y_linear=(data_[0]@weight)+b
            y_hat=1/(1+np.exp(-1*y_linear))
            loss_=-1*(data_[1]*np.log(y_hat)+(1-data_[1])*np.log(1-y_hat))
            loss=np.mean(loss_)
            losses.append(loss)
        loss_val.append(np.mean(losses))
    return loss_train,loss_val,weight,b

def test_acuracy(weight_,bias_):
    X_test_comp=PCA(X_train,X_test,30)
    y_linear=(X_test_comp@weight_)+bias_
    y_hat=1/(1+np.exp(-1*y_linear))
    y_pred=np.where(y_hat>=0.5,1,0)
    matches=0
    # print(y_test)
    for i in range(len(y_pred)):
        if y_test[i]==y_pred[i]:
            matches+=1

    return matches/len(y_pred)




epochs=20
loss_t,loss_v,weight,bias=trainer(epochs,1e-2,1)
print("The test accuracy by the model trained by SGD with 20 epochs and 1e-3 learning rate is = ",test_acuracy(weight,bias))

plt.plot(range(epochs),loss_v,label="validation_loss")
plt.plot(range(epochs),loss_t,label="train_loss")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.title("SGD")
plt.legend()
plt.show()

batch_size=[32,64,128]
lr=[1e-3,1e-2,1e-1]

for i in batch_size:
    for j in lr:
        title="batch_size = "+str(i)+" learning rate = "+str(j)
        loss_t,loss_v,weight,bias=trainer(epochs,j,i)
        print("The test accuracy by the model trained by "+title+" = ",test_acuracy(weight,bias))
        plt.plot(range(epochs),loss_v,label="validation_loss")
        plt.plot(range(epochs),loss_t,label="train_loss")
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.title(title)
        plt.legend()
        plt.show()


lambda_=[1e-2,1e-1,1]
# lr=[1e-3,1e-2,1e-1]
for i in lambda_:
    title="regularization = "+str(i)
    loss_t,loss_v,weight,bias=trainer(epochs,1e-3,64,i)
    print("The test accuracy by the model trained by "+title+" = ",test_acuracy(weight,bias))
    plt.plot(range(epochs),loss_v,label="validation_loss")
    plt.plot(range(epochs),loss_t,label="train_loss")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.title(title)
    plt.legend()
    plt.show()











"""__________________________________________________________QUESTION 6__________________________________________________________"""
# Loading data and converting images into numpy array for further implementation of algorithms

train_dir_path = "Data/emotion_classification/train"

X_train=[]
Y_train=[]
for file in os.listdir(train_dir_path):
    file_path=os.path.join(train_dir_path,file)
    if file_path[-5:-4]=="d":
        Y_train.append(0)
    else:
        Y_train.append(1)
    im=np.array(Image.open(file_path))
    X_train.append(im.ravel())
X_train=np.array(X_train)/255
Y_train=np.expand_dims(np.array(Y_train),axis=1)

test_dir_path = "Data/emotion_classification/test"

X_test=[]
Y_test=[]
for file in os.listdir(test_dir_path):
    file_path=os.path.join(test_dir_path,file)
    if file_path[-5:-4]=="d":
        Y_test.append(0)
    else:
        Y_test.append(1)
    im=np.array(Image.open(file_path))
    X_test.append(im.ravel())
X_test=np.array(X_test)/255
Y_test=np.expand_dims(np.array(Y_test),axis=1)



def PCA(train_data,test_data,num_features):

    """In this PCA code from scratch the top eigen vectors are calculated from traing data and 
        the transformation is done on any data given as argument under test data
    Attributes:
                train_data: eigen vectors are calculated using this data
                test_data: this data is projected to lower dimensions
                num_features: The dimension of the subspace on which data is being projeted 
    Returns:
                compreesed data with lower dimension"""
    
    # import pdb; pdb.set_trace()
    
    if num_features>19:
        raise ValueError("compression features should be less than N-1 for high dimensional PCA i.e. 19")
    train_mean=np.expand_dims(train_data.mean(axis=0), axis=1)
    normalized_data=train_data-train_mean.T
    covariance_mat=(normalized_data@normalized_data.T)/(train_data.shape[0])
    # covariance_mat=np.cov(normalized_data)
    eig_val,eig_vec=np.linalg.eigh(covariance_mat)


    #sort the eigenvalues and eigen vectors in descending order
    idx = eig_val.argsort()[::-1]   
    eigenValues = eig_val[idx]
    
    eigenVectors = eig_vec[:,idx]
    original_eigen_vectors=(1/(np.sqrt(train_data.shape[0]*np.expand_dims(eigenValues[:-1],axis=0))))*((normalized_data.T@eigenVectors)[:,:-1])
    compressed_data=(test_data-train_mean.T)@original_eigen_vectors[:,:num_features]
    return compressed_data

compressed_train_data=PCA(X_train,X_train,12)


class Linear():
    def __init__(self, input_size, output_size):
        # self.weights=np.random.normal(0,1e-3,size=(output_size,input_size))
        sigma = np.sqrt(2 / (input_size + output_size))
        self.weights=np.random.normal(0, sigma, size=(output_size,input_size))
    
    def forward(self,input_x):
        self.input = input_x
        return self.weights@input_x

    def backward_w(self, grad_from_previous):
        return grad_from_previous@(self.input.T)
    
    def backward_x(self,grad_from_previous):
        return self.weights.T@grad_from_previous
    
class ReLu():
    def forward(self,input):
        self.input=input
        out=np.copy(input)
        out[out<0]=0
        return out

    def backward(self,grad_from_previous):
        mask=np.copy(self.input)
        mask=np.where(mask>=0,1,0)
        return mask * grad_from_previous

class MLP_layer2():
    def __init__(self, input_size, output_size,hidden_size):
        self.layer1=Linear(input_size,hidden_size)
        self.relu1=ReLu()
        self.layer2=Linear(hidden_size,output_size)
    
    def forward(self,input_x):
        layer1_linear_out=self.layer1.forward(input_x)
        layer1_relu_out=self.relu1.forward(layer1_linear_out)
        layer2_linear_out=self.layer2.forward(layer1_relu_out)
        return layer2_linear_out
    
    def backward(self,grad_from_previous):
        g2=self.layer2.backward_w(grad_from_previous)
        gx2=self.layer2.backward_x(grad_from_previous)
        gr1=self.relu1.backward(gx2)
        g1=self.layer1.backward_w(gr1)
        return g1,g2
    
def loss_function(input_y,scores):
    ## WRITE CODE HERE  
    return -np.sum(input_y.T * np.log(scores+ 1e-8)) / input_y.shape[0]


def loss_backward(scores,input_y):
    # This part deals with the gradient of the loss w.r.t the output of network
    # for example, in case of softmax loss(-log(q_c)), this part gives grad(loss) w.r.t. q_c
    # pass this to backward_ldata

    ## WRITE CODE HERE    
    N = scores.shape[1]
    return (scores-input_y)/N

# Train the MLP
# creating a function trainer
def trainer_2layer(x,y,h_s,epochs=100,batchsize=32,learning_rate=1e-2,reg_param=1e-4):
    losses=[]
    accuracies=[]
    model=MLP_layer2(x.shape[1],y.shape[1],h_s)
    iter=0
    for i in range(epochs):
        # minibatch=get_minibatch(x,y,batchSize=batchsize)
        # for iter_num,training_data in enumerate(minibatch):
            
        # Write code here for each iteration of training
        # Forward pass
        z=model.forward(x.T)

        max=np.max(z,axis=0)
        z=z-max
        z=np.exp(z)
        deno=np.sum(z,axis=0)
        scores=z/deno
        
        # Backward pass
        grad_from_loss=loss_backward(scores,y.T)
        g1,g2=model.backward(grad_from_loss)

        
        
        # Update weights
        model.layer1.weights=model.layer1.weights-learning_rate*g1
        model.layer2.weights=model.layer2.weights-learning_rate*g2
        
        # Log the training loss value and training accuracy 
        losses.append(loss_function(y,scores))

        # accuracies
        
        train_pred = np.zeros(scores.shape)
        train_pred[np.argmax(scores, axis=0), np.arange(scores.shape[1])] = 1
        diff=np.abs(train_pred-y.T)
        acc=np.count_nonzero(np.sum(diff,axis=0)==0)/len(np.sum(diff,axis=0))
        
        accuracies.append(acc)
        iter=iter+1
            
    return iter,model.layer1.weights,model.layer2.weights,losses,accuracies

# traing the two layer neural network
y_train1 = np.zeros((np.squeeze(Y_train).size, np.squeeze(Y_train).max() + 1))
y_train1[np.arange(np.squeeze(Y_train).size), np.squeeze(Y_train)] = 1

y_test1 = np.zeros((np.squeeze(Y_test).size, np.squeeze(Y_test).max() + 1))
y_test1[np.arange(np.squeeze(Y_test).size), np.squeeze(Y_test)] = 1

compressed_test_data=PCA(X_train,X_test,12)

iter,weights1,weights2,losses,accuracies=trainer_2layer(x=compressed_train_data,y=y_train1,h_s=10,epochs=20,learning_rate=0.2)

plt.plot(range(iter),losses)
plt.xlabel("epochs")
plt.ylabel("loss")

plt.show()

def val_acc(weights1_,weights2_):
    
    relu=ReLu()
    z=weights2_@(relu.forward(weights1_@compressed_test_data.T))
    max=np.max(z,axis=0)
    z=z-max
    z=np.exp(z)
    deno=np.sum(z,axis=0)
    scores=z/deno
    
    train_pred_ = np.zeros(scores.shape)
    train_pred_[np.argmax(scores, axis=0), np.arange(scores.shape[1])] = 1
    
    diff=np.abs(train_pred_-y_test1.T)
    acc=np.count_nonzero(np.sum(diff,axis=0)==0)/len(np.sum(diff,axis=0))
    return acc

print(val_acc(weights1,weights2))




