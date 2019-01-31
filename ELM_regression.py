# -*- coding: utf-8 -*-
"""
Greville and OPIUM method for classifying MNIST from:
J. Tapson and A. van Schaik, 
"Learning the Pseudoinverse Solution to Network Weights"
Neural Networks

Used for Figure 2.

@author: andrevanschaik
"""

from numpy import zeros, random, eye, tanh, dot, reshape, arange, savetxt, loadtxt
from numpy import *
from OPIUM import *
import os.path
import time
import sounddevice as sd
import pyaudio
from pylab import figure, plot, ion
ion()
start_time = time.time()

# Network parameters
size_input=2500
size_hidden=size_input*8   # size of hidden layer
M=zeros((1,size_hidden))  # initial value of linear weights
random_weights=50*(random.rand(size_hidden,size_input)-0.5)
#random_weights=(random.rand(size_hidden,size_input))/320.0
N_train = 100000
N_test = 3000
# to save the weights and reload them to use the same set in multiple runs, use:
savetxt('randw.dat',random_weights)
random_weights=loadtxt('randw.dat')

# Initialisation of signal matrices
h=zeros((size_hidden,1))
e=zeros((1,1))
Theta=eye(size_hidden)

angle_array = (-90, -75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75, 90)  # 13 angles

k =0 
p = 0
cross_corr = zeros((11280, 2501))
#arr = arange(5*20*13*10)
arr = arange(11280)
arr_train = random.shuffle(arr)
count_n = 0

cross_corr_cut = zeros((30,30))
cross_corr_reshape = zeros(2500)
cross_corr_dia = zeros(100)
cross_corr_dia_1 =zeros(50)
############################### load training data sets################################### 
n=0
for i in range (1,4):
    data_dir = ".\sound_localisation\sound_data_base_real_4\sound_localization_data_%d" %i
    for j in range (0,20):
        data_set_dir = "%s\data_set_%d"  % (data_dir,j)
        #print ("%s" %data_set_dir)
        for s in range (10):
            for index, item in enumerate (angle_array):
                data_set_angle_dir = "%s\cross_matrix%d_%s.txt" %(data_set_dir,s,item)
                #print("%s" %data_set_angle_dir)
                if os.path.exists(data_set_angle_dir):
                    count_n = count_n +1 
                    corr_array = loadtxt(data_set_angle_dir)
                    cross_corr_cut = corr_array[0:50,50:100]/320.0
                    cross_corr_reshape = cross_corr_cut.reshape(2500)
                    #print (corr_array)
                    label_vector = zeros((1,1))
                    label_vector = index
                    n=n+1
                    ###################random shuffle data set#########################
                    cross_corr[p,0] = index
                    cross_corr[p,1:2501] =cross_corr_reshape
                    p = p + 1                     
                    ##################################################################

for l in range (8000):
            corr_array_train = cross_corr[arr[l],1:]
            #label_vector = zeros((1,1))
            label_vector = cross_corr[arr[l],0]
            h = tanh (dot(random_weights,reshape(corr_array_train,(size_input,1))))
            k = k +1
            y = dot(M,h)                
            e = reshape(label_vector,(1,1))-y
        
######################### Choice between the Greville and OPIUM method#############################    
            #OPIUM(h,e,M,Theta)
            OPIUMl(h,e,M,1)

errors=0
qlist=[]
Y=zeros((1,N_train))
count_n = 0
###############################   Load the testing datasets###################################
ang_rec = zeros((size_input,1))

for l in range (8000,11280):
            ang_rec = cross_corr[arr[l],1:]
            #print (corr_array)
            label_vector = zeros((1,1))
            #label_vector[cross_corr[arr[l],0]] = 1.0
            label_vector = cross_corr[arr[l],0]
            #print label_vector.T 
            #print (label_vector)        
            h = tanh (dot(random_weights,reshape(ang_rec,(size_input,1)))) 
            y = dot(M,h)                 
            D = y.argmax()
            #Y[D,q] = 1
            #if (index!=D):
            print ("The location of the speeker %s is at %s degree" %(angle_array[int(cross_corr[arr[l],0])], angle_array[int(y)]))

            '''
            if (cross_corr[arr[l],0] != D):
                errors = errors+1
                #qlist.append(q)
                print (" error %s is %s" %(cross_corr[arr[l],0],D) )
            else:
                print ("The location of the speeker is at %s degree" %item)
             '''   
'''
errors = 0
ang_rec = zeros((size_input,1))
#for l in range (10001,13000):
for l in range (5000,7000):
            ang_rec = cross_corr[arr[l],1:]
            #print (corr_array)
            label_vector = zeros((13,1))
            label_vector[cross_corr[arr[l],0]] = 1.0
            #print label_vector.T 
            #print (label_vector)        
            h = tanh (dot(random_weights,reshape(ang_rec,(size_input,1)))) 
            y = dot(M,h)                 
            D = y.argmax()
            #Y[D,q] = 1
            #if (index!=D):
            if (cross_corr[arr[l],0] != D):
                errors = errors+1
                #qlist.append(q)
                print (" error %s is %s" %(cross_corr[arr[l],0],D) )
            else:
                print ("The location of the speeker is at %s degree" %item)

#print qlist
#print random_weights
#ion()
#for i in range(13):
#    figure
#    plot(eval('W'+str(i)))

end_time = time.time()
print("Elapsed time was %g seconds" % (end_time - start_time))
'''






