import numpy as np
import matplotlib.pyplot as plt
import librosa
import os
from PIL import Image
seed = 26
np.random.seed(seed)


"""__________________________________________________________QUESTION 5__________________________________________________________"""


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




"""compression of traning and test data to ! dimension"""
reduced_train=LDA_after_PCA(X_train,14)
reduced_test=LDA_after_PCA(X_test,14)

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


"""Below code snippet is iteratting over several threshold values to calculate the threshold
that gives the best accuracy on training set"""

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



"""__________________________________________________________QUESTION 6__________________________________________________________"""


features=128

# step1 - converting a wav file to numpy array and then converting that to mel-spectrogram
audio_clean, sample_rate_clean= librosa.load("clean.wav", sr=16000)
audio_noisy, sample_rate_noisy= librosa.load("noisy.wav", sr=16000)
# step2 - converting audio np array to spectrogram
clean = librosa.feature.melspectrogram(y=audio_clean,
                                        sr=sample_rate_clean, 
                                            n_fft=2048, 
                                        hop_length=160, 
                                            win_length=320, 
                                            window='hann', 
                                            center=True, 
                                            pad_mode='reflect', 
                                            power=2.0,
                                     n_mels=features).astype(np.float64)

noisy = librosa.feature.melspectrogram(y=audio_noisy,
                                        sr=sample_rate_noisy, 
                                            n_fft=2048, 
                                        hop_length=160, 
                                            win_length=320, 
                                            window='hann', 
                                            center=True, 
                                            pad_mode='reflect', 
                                            power=2.0,
                                     n_mels=features)


## PART 1

def whitening_transform(train_data,test_data,compression_dim):

    """Applies whitening transfor to the data. It first calculates the eigen vector and eigen values from clean data and then transforms the data
    given under data argumen
    
    arguments : data - data on which transformation is applied
                compression_dim - finall dimension of the whitened data
    returns whitened data and covariance matrix of whitened data"""

    normalized_data=train_data-np.expand_dims(train_data.mean(axis=1),axis=1)

    covariance_mat=(normalized_data@normalized_data.T)/train_data.shape[1]+1e-5*np.eye(128,128)
    eig_val,eig_vec=np.linalg.eigh(covariance_mat)

    #sort the eigenvalues and eigen vectors in descending order
    idx = eig_val.argsort()[::-1]   
    eigenValues = eig_val[idx]
    eigenVectors = eig_vec[:,idx]


    PCA_projection_mat=eigenVectors[:,:compression_dim]

    top_eig_mat=np.diag(eigenValues[:compression_dim]**(-0.5))

    normalized_test=test_data-np.expand_dims(train_data.mean(axis=1),axis=1)
    whitened_data=(top_eig_mat)@PCA_projection_mat.T@normalized_test
    return whitened_data , (whitened_data@whitened_data.T)/whitened_data.shape[1]


## priniting covariance matrix of clean and noisy data

print(np.diag(whitening_transform(clean,noisy,features)[1]))
print(np.diag(whitening_transform(noisy,clean,features)[1]))
print(np.linalg.norm(whitening_transform(clean,noisy,features)[1]-np.eye(128,128)))


## Part 2





def k_means(data):
    a=np.random.randint(0,data.shape[0])
    b=np.random.randint(0,data.shape[0])

    culster_1_cen=data[a]
    culster_2_cen=data[b]
    
    for m in range(1):
        clusters=[]
        for i in range(data.shape[0]):
            if (np.linalg.norm(data[i]-culster_1_cen))>=(np.linalg.norm(data[i]-culster_2_cen)):
                clusters.append(1)
            else:
                clusters.append(2)
        cluster_1_sum=0
        cluster_2_sum=0
        cluster1_cov=0
        cluster2_cov=0
        for i,j in zip(data,clusters):
            if j==1:
                cluster_1_sum=cluster_1_sum+i
                cluster1_cov=cluster1_cov+(np.expand_dims(i,axis=1)@np.expand_dims(i,axis=1).T)
            else:
                cluster_2_sum=cluster_2_sum+i
                cluster2_cov=cluster2_cov+(np.expand_dims(i,axis=1)@np.expand_dims(i,axis=1).T)
        culster_1_cen=cluster_1_sum/clusters.count(1)
        culster_2_cen=cluster_2_sum/clusters.count(2)
        cluster1_cov=cluster1_cov/clusters.count(1)
        cluster2_cov=cluster2_cov/clusters.count(2)
    return [culster_1_cen,culster_2_cen],[cluster1_cov,cluster2_cov]


def softmax(arr):
    max=np.max(arr)
    arr=arr-max
    deno=np.exp(arr)
    return deno/deno.sum()


def gmm_em(data, num_gauss, k_means_:bool, eps=1e-6):

    if k_means_==False:
        guesses = np.random.randint(0, num_gauss, size=(data.shape[0], ))
        means, cov_matrices = [], []
        for i in range(num_gauss):
            mask = (guesses == i)
            means.append(data[mask].mean(axis=0).astype(np.float64))
            covs = ((data[mask].T @ data[mask]) / np.sum(mask) )+ eps*np.eye(data.shape[1])
            cov_matrices.append(covs)
    else:
        means, cov_matrices = k_means(data)
        for i in range(len(cov_matrices)):
            cov_matrices[i]=cov_matrices[i]+ eps*np.eye(data.shape[1])
    
    mixing_coeff = np.random.randint(10, size=num_gauss).astype(np.float64)
    mixing_coeff = np.exp(mixing_coeff)
    mixing_coeff = mixing_coeff/mixing_coeff.sum()

    log_liklihood=0
    for i in range(data.shape[0]):
        interior_val=[]
        for k in range(num_gauss):
            mean=means[k]
            cova=cov_matrices[k]
            val=np.log(mixing_coeff[k])+((-0.5)*np.linalg.slogdet(cova)[1])+((-0.5)*((np.expand_dims(data[i],axis=0)-mean)@np.linalg.inv(cova)@(np.expand_dims(data[i],axis=0)-mean).T))
            interior_val.append(val.astype(np.float64))
        max_i=np.max(interior_val)
        final_val=0
        for q in interior_val:
            final_val=(final_val+np.exp(q-max_i)).astype(np.float64)
        final_val=np.log(final_val)
        final_val=final_val+max_i

        log_liklihood=log_liklihood+final_val

    log_lik_his=[]
    log_lik_his.append(log_liklihood.item())

    for m in range(20):
        ## E- Step
        responsibility_mat=[]
        for i in range(data.shape[0]):
            arr=[]
            for k in range(num_gauss):
                mean=means[k]
                cova=cov_matrices[k]
                val=np.log(mixing_coeff[k])+((-0.5)*np.linalg.slogdet(cova)[1])+((-0.5)*((np.expand_dims(data[i],axis=0)-mean)@np.linalg.inv(cova)@(np.expand_dims(data[i],axis=0)-mean).T))
                arr.append(val.item())

            arr = softmax(arr)
            responsibility_mat.append(arr)

        responsibility_mat = np.squeeze(np.array(responsibility_mat, dtype=np.float64))

        ## M- Step
        N_k=np.sum(responsibility_mat,axis=0)

        means=[]
        for k in range(num_gauss):
            means.append(np.sum(data * (responsibility_mat[:,k][:,None]), axis=0)/ N_k[k])

        mixing_coeff=N_k/data.shape[0]

        cov_matrices=[]
        for k in range(num_gauss):
            temp=0
            for i in range(data.shape[0]):
                temp=temp+responsibility_mat[i,k]*((np.expand_dims((data[i]-means[k]),axis=1))@(np.expand_dims((data[i]-means[k]),axis=1)).T)

            temp = temp/N_k[k]
            cov_matrices.append(temp+ eps*np.eye(data.shape[1]))

        log_liklihood = 0
        for i in range(data.shape[0]):
            interior_val=[]
            for k in range(num_gauss):
                mean=means[k]
                cova=cov_matrices[k]
                val=np.log(mixing_coeff[k]) + ((-0.5)*np.linalg.slogdet(cova)[1]) + ((-0.5)*((np.expand_dims(data[i],axis=0)-mean)@np.linalg.inv(cova)@(np.expand_dims(data[i],axis=0)-mean).T))
                interior_val.append(val.item())

            max_i=np.max(interior_val)

            final_val=0
            for q in interior_val:
                final_val = final_val + np.exp(q-max_i)

            final_val=np.log(final_val)
            final_val = final_val + max_i

            log_liklihood=log_liklihood+final_val

        log_lik_his.append(log_liklihood)


    return log_lik_his,means,cov_matrices,mixing_coeff
    


K=2 ## this is the no of gaussians you like to fit on the given data

log_lik_his,means,cov_matrices,mixing_coeff=gmm_em(clean.T, K, False, eps=1e-6)
plt.scatter(range(21), log_lik_his,label='random initialisation',color='hotpink')

log_lik_his,means,cov_matrices,mixing_coeff=gmm_em(clean.T, K, True, eps=1e-6)
plt.scatter(range(21), log_lik_his,label='k-means initialisation',color='teal')

plt.xlabel("iterations")
plt.ylabel("log liklihood")
plt.legend(loc="lower right")
plt.show()
