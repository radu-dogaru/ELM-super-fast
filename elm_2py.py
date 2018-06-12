# -*- coding: utf-8 -*-
"""
Last update: June 6, 2018 
@authors: radu.dogaru@upb.ro ioana.dogaru@upb.ro
 
Implements ELM training using datasets available in Matlab format 
Similar to the Octave / Matlab implementation 
Tested under Python 3.6 (Anaconda 5.1)

Software supporting the article: 
[1] Radu Dogaru*, Ioana Dogaru, "Optimized Extreme Learning Machine for Big Data
Applications using Python", in COMM-2018, The 12th International Conference 
on Communications, Bucharest, Romania, 14-16 June 2018. 

Please cite the above article in works where this software is used
"""

import numpy as np
import scipy.io as sio
import scipy.linalg
import time as ti 


def hidden_nonlin(hid_in, tip):
# implementation of the hidden layer 
# additional nonlinearitys may be added 
    if tip==0: 
        # sigmoid 
        H=np.tanh(hid_in)        
    elif tip==1:
        # linsat 
        H=abs(1+hid_in)-abs(1-hid_in)
    elif tip==2:
        # ReLU
        H=abs(hid_in)+hid_in
    elif tip==3:
        # see [1] - very well suited for emmbeded systems 
        H=abs(hid_in)
    elif tip==4:
        H=np.sqrt(hid_in*hid_in+1)
        # multiquadric 
    return H
        

def elmTrain_optim(X, Y, h_Neurons, C , tip):
# Training phase - floating point precision (no quantization)
# X - Samples (feature vectors) Y - Labels
      Ntr = np.size(X,1)
      in_Neurons = np.size(X,0)
      classes = np.max(Y)
      # transforms label into binary columns  
      targets = np.zeros( (classes, Ntr), dtype='int8' )
      for i in range(0,Ntr):
          targets[Y[i]-1, i ] = 1
      targets = targets * 2 - 1
      
      #   Generate inW layer  
      rnd = np.random.RandomState()
      inW=-1+2*rnd.rand(h_Neurons, in_Neurons).astype('float32')
      #inW=rnd.randn(nHiddenNeurons, nInputNeurons).astype('float32')
      
      #  Compute hidden layer 
      hid_inp = np.dot(inW, X)
      H=hidden_nonlin(hid_inp,tip)
      
      # Moore - Penrose computation of output weights (outW) layer 
      outW = scipy.linalg.solve(np.eye(h_Neurons)/C+np.dot(H,np.transpose(H)), np.dot(H,np.transpose(targets)))     
      
      return inW, outW 

# implements the ELM training procedure with weight quantization       
def elmTrain_fix( X, Y, h_Neurons, C , tip, ni):
# Training phase - emulated fixed point precision (ni bit quantization)
# X - Samples (feature vectors) Y - Labels
# ni - number of bits to quantize the inW weights 
      Ntr = np.size(X,1)
      in_Neurons = np.size(X,0)
      classes = np.max(Y)
      # transforms label into binary columns  
      targets = np.zeros( (classes, Ntr), dtype='int8' )
      for i in range(0,Ntr):
          targets[Y[i]-1, i ] = 1
      targets = targets * 2 - 1
      
      #   Generare inW 
      #   Generate inW layer  
      rnd = np.random.RandomState()
      inW=-1+2*rnd.rand(h_Neurons, in_Neurons).astype('float32')
      #inW=rnd.randn(nHiddenNeurons, nInputNeurons).astype('float32')
      Qi=-1+pow(2,ni-1) 
      inW=np.round(inW*Qi)
      
      #  Compute hidden layer 
      hid_inp = np.dot(inW, X)
      H=hidden_nonlin(hid_inp,tip)
      
      # Moore - Penrose computation of output weights (outW) layer 
      outW = scipy.linalg.solve(np.eye(h_Neurons)/C+np.dot(H,np.transpose(H)), np.dot(H,np.transpose(targets)))     
      
      return inW, outW 
      

def elmPredict_optim( X, inW, outW, tip):
# implements the ELM predictor given the model as arguments 
# model is simply given by inW, outW and tip 
# returns a score matrix (winner class has the maximal score)

      hid_in=np.dot(inW, X)
      H=hidden_nonlin(hid_in,tip)
      score = np.transpose(np.dot(np.transpose(H),outW))
      return score 
        
# ======================================================
#  RUNNING EXAMPLE 
#================================================================================
# parameters 
nume='optd64' # Database (Matlab format - similar to what is supported by the LIBSVM library)
#nume='mnist' # MNIST dataset 
nr_neuroni=2000 # Proposed number of neurons on the hidden layer 
C=0.1 # Regularization coefficient C  
tip=3 # Nonlinearity of the hidden layer  
nb_in=2;  # 0 = float; x - represents weights on a finite x number of bits 
nb_out=0; # same as above but for the output layer

#===============  TRAIN DATASET LOADING ==========================================
# converts into 'float32' for faster execution 
t1 = ti.time()
db=sio.loadmat(nume+'_train.mat')
Samples=db['Samples'].astype('float32')
Labels=db['Labels'].astype('int8')
clase=np.max(Labels)
trun = ti.time()-t1
print(" load train data time: %f seconds" %trun)

#================= TRAIN ELM =====================================================
t1 = ti.time()
if nb_in>0:
    inW, outW = elmTrain_fix(Samples, np.transpose(Labels), nr_neuroni, C, tip, nb_in)
else:
    inW, outW = elmTrain_optim(Samples, np.transpose(Labels), nr_neuroni, C, tip)
trun = ti.time()-t1
print(" training time: %f seconds" %trun)

# ==============  Quantify the output layer ======================================
Qout=-1+pow(2,nb_out-1)
if nb_out>0:
     O=np.max(np.abs(outW))
     outW=np.round(outW*(1/O)*Qout)

#================= TEST (VALIDATION) DATASET LOADING 
t1 = ti.time()
db=sio.loadmat(nume+'_test.mat')
Samples=db['Samples'].astype('float32')
Labels=db['Labels'].astype('int8')
n=Samples.shape[0]
N=Samples.shape[1]
trun = ti.time()-t1
print( " load test data time: %f seconds" %trun)
#====================== VALIDATION PHASE (+ Accuracy evaluation) =================
t1 = ti.time()
scores = elmPredict_optim(Samples, inW, outW, tip)
trun = ti.time()-t1
print( " prediction time: %f seconds" %trun)

# CONFUSION MATRIX computation ==================================
Conf=np.zeros((clase,clase),dtype='int16')
for i in range(N):
    # gasire pozitie clasa prezisa 
    ix=np.nonzero(scores[:,i]==np.max(scores[:,i]))
    pred=int(ix[0])
    actual=Labels[0,i]-1
    Conf[actual,pred]+=1
accuracy=100.0*np.sum(np.diag(Conf))/np.sum(np.sum(Conf))
print("Confusion matrix is: ")
print(Conf)
print("Accuracy is: %f" %accuracy)
print( "Number of hidden neurons: %d" %nr_neuroni)
print( "Hidden nonlinearity (0=sigmoid; 1=linsat; 2=Relu; 3 - ABS; 4- multiquadric): %d" %tip)
    
#====================================================================================   

'''
Running example (on MNIST)  with 2 bits per weights in the input layer 
Using MKL-NUMPY / CPU: Intel Core-I7 6700HQ (4-cores @ 2.6Ghz)

 load train data time: 1.328532 seconds
 training time: 25.102763 seconds
 load test data time: 0.314851 seconds
 prediction time: 1.308466 seconds
Confusion matrix is: 
[[ 970    1    1    0    0    1    3    1    2    1]
 [   0 1126    2    1    1    0    2    0    3    0]
 [   6    0  987   10    3    0    2    8   14    2]
 [   0    0    2  986    0    6    0    6    6    4]
 [   1    0    2    0  961    0    5    2    2    9]
 [   3    0    0    9    1  866    8    2    1    2]
 [   5    2    1    0    4    4  934    0    8    0]
 [   0    9   12    3    2    1    0  986    3   12]
 [   3    0    2    9    2    2    2    5  945    4]
 [   5    5    3    9   11    5    0    6    1  964]]
Accuracy is: 97.250000
Number of hidden neurons: 8000
Hidden nonlinearity (0=sigmoid; 1=linsat; 2=Relu; 3 - ABS; 4- multiquadric): 3
inW
Out[119]: 
array([[ 0.,  1., -1., ..., -0., -0., -0.],
       [-0.,  1., -0., ...,  1.,  1.,  0.],
       [ 0., -1., -0., ...,  0.,  0., -1.],
       ...,
       [-1., -1.,  0., ..., -0., -1., -1.],
       [-1., -1., -0., ..., -0., -0.,  1.],
       [ 0., -0., -1., ..., -1., -1.,  0.]], dtype=float32)
'''
    