import numpy as np
import matplotlib.pyplot as plt
import wave
import struct
import sys
from scikits.audiolab import *
import random
from datetime import datetime
import operator
import scipy
from scipy import signal
import math
from pylab import*
import cmath
import operator
from tempfile import TemporaryFile

def ground_truth(labfile):
	dict = {'N':None,'C':0,'C#':1,'D':2,'D#':3,'E':4,'F':5,'F#':6,'G':7,'G#':8,'A':9,'A#':10,'B':11,
	'c':12,'c#':13,'d':14,'d#':15,'e':16,'f':17,'f#':18,'g':19,'g#':20,'a':21,'a#':22,'b':23}

	ground = np.zeros(1250)
	list = []
	with open(labfile,'r') as g:
	            f=g.readlines()
	            for line in f:
	        		a=(line.split(' '))
	        		list.append(a)
	        		ground[np.floor(float(a[0])*86.13):np.floor(float(a[1])*86.13)] = dict[a[2].rstrip("\n")]
	        		if float(a[1])>15:
	        			break
	return ground

def generate_template():
	b = np.zeros(288).reshape(12,24)

	a = [0,0,0,1,0,0,0,1,0,0,1,0]
	b[:,0] = a

	for i in range(1,12):
		b[:,i] = np.roll(b[:,i-1],1)

	minor = [0,0,0,1,0,0,1,0,0,0,1,0]
	b[:,12] = minor

	for i in range(13,24):
		b[:,i] = np.roll(b[:,i-1],1)
	
	return b

def viterbi(chrm):

	temp = generate_template()
	#print "template shape is",shape(temp)
	trans = np.dot(temp.T,temp)

	trans = np.ones(shape(trans))
	trans = trans + np.eye(24)*0.1
	#print trans
	print "transition matrix shape is",shape(trans)


	V = np.zeros(shape(chrm))
	path = np.zeros(shape(chrm))


	print "shape of V is",shape(V)

	#same prior probabilities
	initial = 1.00

	#initilization
	V[:,0] = np.log(chrm[:,0]) + np.log(initial)
	#V[:,0] = np.zeros(b)

	#chrm = chromgram

	for t in range(1,col):
		for j in range(b):
			sam = [(V[i,t-1]+np.log(trans[i,j])+np.log(chrm[j,t]),i) for i in range(b)]
			#find index and value that maximize sum
			(prob,state) = max(sam)
			V[j,t] = prob
			path[j,t-1] = state

	#print path

	return V,path




"""
Read chroma file and plot it
"""

chroma = np.load('chroma1.npy')
subplot(3,1,1)
title('Chroma')

im = imshow(chroma,aspect='auto',interpolation='nearest',origin='lower',extent=[0,15,0,24])
yticks([1,3,5,7,9,11,13,15,17,19,21,23], ['A','A#','B','C','C#','D','D#','E','F','F#','G','G#'])


"""
Read the Distace matrix file and plot it
"""
chrm = np.load('tempogram1.npy')
b,col = shape(chrm)

whereAreNaNs = np.isnan(chrm);
chrm[whereAreNaNs] = 0;

subplot(3,1,2)
title('Templates')

im = imshow(chrm,aspect='auto',interpolation='nearest',origin='lower',extent=[0,15,0,24])
yticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23], 
	['C','C#','D','D#','E','F','F#','G','G#','A','A#','B','c','c#','d','d#','e','f','f#','g','g#','a','a#','b'])


"""
Viterbi Algorithm
"""

V,path = viterbi(chrm)



Vnorm = np.zeros(shape(V))

#normalize
for i in range(col):
	Vnorm[:,i] = V[:,i]/(sum(V[:,i]))

Vm = np.zeros(shape(V))


#median filter
for i in range(b):
	for j in range(col):
		Vm[i,j] = np.mean(V[i,max(0,j-100):min(col,j+100)])


#picking chords with max value
inde = np.argmax(Vm,axis=0)
final_path = np.zeros(col)

for i in range(col):
	final_path[i] = path[inde[i],i]


ground = ground_truth('Let It Be_new.lab')

print "ground truth length is",len(ground)


#plot detected and ground thruth chords
subplot(3,1,3)
title('Chords')
yticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23], 
	['C','C#','D','D#','E','F','F#','G','G#','A','A#','B','c','c#','d','d#','e','f','f#','g','g#','a','a#','b'])
l = plot(ground,'ro',alpha=0.3)
setp(l, 'markersize', 15)



plot(final_path,'ko',alpha=0.2)
yticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23], 
	['C','C#','D','D#','E','F','F#','G','G#','A','A#','B','c','c#','d','d#','e','f','f#','g','g#','a','a#','b'])
xlim(0,col)
xticks(np.arange(0,col,172),[0,2,4,6,8,10,12,14])
show()



print 'end'
