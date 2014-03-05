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
print shape(chrm)
b,col = shape(chrm)

whereAreNaNs = np.isnan(chrm);
chrm[whereAreNaNs] = 0;

chrm_new = np.zeros(shape(chrm))

#median filter
#for mean, change np.median to np.mean

for i in range(b):
	for j in range(col):
		chrm_new[i,j] = np.mean(chrm[i,max(0,j-10):min(col,j+10)])


#print chrm[0,col:col+100]
#print chrm_new


#comment out the line below if you use median filter
#chrm_new = chrm


subplot(3,1,2)
title('Templates')

im = imshow(chrm_new,aspect='auto',interpolation='nearest',origin='lower',extent=[0,15,0,24])
yticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23], 
	['C','C#','D','D#','E','F','F#','G','G#','A','A#','B','c','c#','d','d#','e','f','f#','g','g#','a','a#','b'])


"""
Chord detection using max function
"""
chrm_max = np.zeros(shape(chrm_new))
chrm_array = np.zeros(col)

for i in range(col):
	maxl = chrm_new.argmax(axis=0)
	chrm_max[maxl[i],i] = 1

chrm_array = chrm_new.argmax(axis=0)

ground = ground_truth('Let It Be_new.lab')

print "ground truth length is",len(ground)



subplot(3,1,3)


title('Chords')
yticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23], 
	['C','C#','D','D#','E','F','F#','G','G#','A','A#','B','c','c#','d','d#','e','f','f#','g','g#','a','a#','b'])
l = plot(ground,'ro',alpha=0.3)
setp(l, 'markersize', 15)


plot(chrm_array,'ko',alpha=0.2)
yticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23], 
	['C','C#','D','D#','E','F','F#','G','G#','A','A#','B','c','c#','d','d#','e','f','f#','g','g#','a','a#','b'])
xlim(0,col)
xticks(np.arange(0,col,172),[0,2,4,6,8,10,12,14])
show()
 



print 'end'
