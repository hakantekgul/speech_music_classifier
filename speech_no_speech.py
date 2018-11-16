# import the necessary package dependencies
import numpy as np
import scipy
from scipy.io.wavfile import read as wavread
from scipy import signal
from scipy.fftpack import fft, fftfreq, ifft
import scipy.io
import matplotlib
import matplotlib.pyplot as plt
import os
from PIL import Image

# define a PCA function to use (Lecture 5 - Slide 17)
# COV = (1/K)(X - X_mean).(X - X_mean)^T
def PCA(X,n): 
	X_2 = X - np.mean(X,axis=1,keepdims=True)    							# removed the mean
	COV = (X_2.dot(X_2.T))/(X_2.shape[1]-1)        							# computed the covariance matrix  
	eigenvalues, eigenvecs = scipy.sparse.linalg.eigsh(COV,k=n)  		    # Got the eigenvectors and eigenvalues
	W = np.diag(1./(np.sqrt(eigenvalues))).dot(eigenvecs.T)                 # Lecture 5 - Slide 18 --> W = diag(eigenvalues^-1)*U^T
	return W

def gauss_classifier(X):
        N = X.shape[1]
        # get mean and covariances of data
        Mean = np.mean(X,axis=1,keepdims=True)
        Cov = (X-Mean).dot((X-Mean).T)
        Cov /= (N-1)
        # return both of them as a dict 
        g_dict = {'mean':Mean, 'cov':Cov}
        return g_dict

def log_prob(X,g):
    M = X.shape[0]
    X_2 = X - g['mean']
    IC = np.linalg.pinv(g['cov'])
    probs = (-1*np.log(np.linalg.det(IC)) + M*np.log(2*np.pi) + np.sum((IC).dot(X_2)*X_2,axis=0))*(-1/2)
    return probs

np.random.seed()
speech_path = 'SpeechMusic/speech'
music_path = 'SpeechMusic/music'
diff = '340.wav'
data_speech = []
data_music = []
sampling_rate = 22050
file_len = 330750
num_file = 60
num_test = 0.1 * num_file
pca_dims = 64

mix = np.arange(0, num_file, 1) # for random sampling of data
np.random.shuffle(mix)

# First, load the speech dataset
for file in os.listdir(speech_path):
    path = os.path.join(speech_path,file)
    if diff in path:
        continue
    fs,speech = wavread(path)
    speech = speech.astype(float)
    data_speech.append(speech)

# Next, load the music dataset    
for file in os.listdir(music_path):
    path = os.path.join(music_path,file)
    fs,music = wavread(path)
    music = music.astype(float)
    data_music.append(music)
data_music = np.array(data_music)
data_speech = np.array(data_speech)

print(data_speech.shape)
f,t,Zxx_speech = signal.stft(data_speech, fs = sampling_rate, window = 'hann', nperseg = 1024, noverlap=768)
Zxx_speech = np.abs(Zxx_speech)
Zxx_speech = np.log(Zxx_speech)

f,t,Zxx_music = signal.stft(data_music, fs = sampling_rate, window = 'hann', nperseg = 1024, noverlap=768)
Zxx_music = np.abs(Zxx_music)
Zxx_music = np.log(Zxx_music)

Train_speech = np.hstack(Zxx_speech[mix[6:]]) # random 90% of speech data sampled
Train_music = np.hstack(Zxx_music[mix[6:]]) # random 90% of music data sampled
Test_speech = np.hstack(Zxx_speech[mix[:6]]) # random remaining 10% is for testing
Test_music = np.hstack(Zxx_music[mix[:6]])

X_train = np.hstack((Train_speech,Train_music)) # stack'em
X_test = np.hstack((Test_speech,Test_music)) #stack'em

# Create labels for testing
# 1 is speech and 0 if music
speech_labels = np.ones(Test_speech.shape[1])
music_labels = np.zeros(Test_music.shape[1])
Y_test = np.hstack((speech_labels,music_labels))

W = PCA(X_train,pca_dims)

Z_speech = W.dot(Train_speech - np.mean(Train_speech,axis=1,keepdims=True))
gauss_speech = gauss_classifier(Z_speech)

Z_music = W.dot(Train_music - np.mean(Train_music,axis=1,keepdims=True))
gauss_music = gauss_classifier(Z_music)

# function to compute classification accuracy
def speech_accuracy(pred,Y_test):
    test_ind = np.where(Y_test == 1)[0] # get the indices of data that has speech label
    pred_num = np.sum(pred[test_ind] == Y_test[test_ind]) #number of correct predictions
    acc = (pred_num / test_ind.shape[0])*100
    return acc
def one_second_result(probs,fps):
    probs = np.convolve(probs,np.ones(fps))
    return probs

Z_test = W.dot(X_test - np.mean(X_test,axis=1,keepdims=True))
# Compute likelihoods for speech and music
speech_probs = log_prob(Z_test,gauss_speech)
music_probs = log_prob(Z_test,gauss_music)
probs = np.hstack((speech_probs,music_probs))

# According to Piazza, we get the 1-sec probs
speech_probs = one_second_result(speech_probs,fps=26)
music_probs = one_second_result(music_probs,fps=26)
# get predictions
result = (speech_probs > music_probs).astype(int)


print('The accuracy of the speech classifier is: %f' %(speech_accuracy(result,Y_test)))