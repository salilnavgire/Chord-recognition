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


def read_audio(filename):
	spf = wave.open(filename,'r')
	signal = spf.readframes(-1)
	signal = np.fromstring(signal, 'Int16')
	p = spf.getnchannels()
	f = spf.getframerate()
	sound_info = np.zeros(len(signal),dtype=float)
	signal = signal.astype(np.float)
	sound_info = signal/max(signal)

	#sound_info = sound_info[1:len(sound_info):2]
	if p==2:
		sound_info = scipy.signal.decimate(sound_info,2)

	return p ,f , sound_info


def spectrogram(sound_info,f,nfft,hop):
	Pxx, freqs, bins, im = specgram(sound_info, Fs = f, NFFT = nfft, noverlap=nfft-hop, scale_by_freq=True,sides='default')
	return Pxx, freqs, bins, im


def hz_to_oct(freq):
	#A6 = 27.5*(2^6)
	#A0 = 27.5
	#A4 = 440
	fmin = 27.5
	b = 24
	return np.log2(freq/fmin)*b


def oct_to_hz(oct):
	fmin = 27.5
	b = 24.0
	return fmin*(2**(oct/b))


def generate_filterbank(NFFT,fs,b,z):
	#b is bins per octave
	#z is number of octaves
	#b = 24
	#z = 6
	#fs(downsampled) = 44100/4 = 11025

	octmax = b*z

	octpts = np.arange(-1,octmax+1)
	#print 'octpts',octpts
	#print len(octpts)

	ctrfrq = oct_to_hz(octpts)
	#print "ctrfrq",ctrfrq
	#print len(ctrfrq)

	ctrrep = np.floor((NFFT+2)*ctrfrq/((fs/2)))
	#print "ctrrep",ctrrep
	#print len(ctrrep)

	bank = np.zeros([len(octpts)-2,NFFT/2+1],dtype=float)

	for j in xrange(0,len(octpts)-2):
		y = np.hamming(ctrrep[j+2]-ctrrep[j])
		area = trapz(y, dx=5)
		if area==0:
			area=1
		y2 = (y/area)
		bank[j,ctrrep[j]:ctrrep[j+2]] = y2
		#plot(bank[j,:])

	#show()
	return bank


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



"""
Read input audio wav file
"""
p, f, sound_info_old = read_audio('Let It Be.wav')
print "frequency is",f
print "channels are",p
print len(sound_info_old)
#30 seconds is 1323000
sound_info_deci = scipy.signal.decimate(sound_info_old,4)
print len(sound_info_deci)
#165375 is 15 sec 330750 is 30 sec for downsampled signal
sound_info = sound_info_deci[0:165375]
print "length of audio is ",len(sound_info)
f = f/4

#plot(sound_info)
#show()


"""
Compute spectrogram
"""

Pxx, freqs, bins, im = spectrogram(sound_info, f, 6000, 128)
print "shape of Spectrogram is",shape(Pxx)
#plot(Pxx,freqs)
#show()


"""
Generate CQ filterabank
"""

bank = generate_filterbank(NFFT=6000,fs=f,b=24,z=6)
print "shape of bank",shape(bank)
#im = imshow(bank,aspect='auto',interpolation='nearest',origin='lower')
#show()


"""
Generate CQ spectrogram as the dot product of spectrogram and CQ filterabank
"""

sal = np.dot(bank,Pxx)
b,col = shape(sal)

#Replace 0 by 1 in matrix before taking log
for i in range(0,len(sal)):
	for j in range(0,len(sal[0])):
		if sal[i][j] == 0:
			sal[i][j]+=1

sal = 10*np.log10(sal)
#Normalize
salm = np.zeros(shape(sal))
salm = (sal - sal.min())/(sal.max()-sal.min())
#salm = sal

"""
for i in range(col):
	salm[:,i] = sal[:,i]/sum(sal[:,i])
"""

print "shape of CQ Spectrogram is",shape(salm)
subplot(3,1,2)
title('CQ spectrogram (dB)')
#xlim(0,(len(sound_info)/11025.00))
im = imshow(salm,aspect='auto',interpolation='nearest',origin='lower')
xticks(np.arange(0,col,172),[0,2,4,6,8,10,12,14])

#show()


"""
Generate Chroma
"""
row,col = shape(salm)
#print row
#print col
b = 24

chrm = np.zeros(b*col).reshape(b,col)

for i in range(col):
	c = salm[:,i]
	for j in range(b):
		chrm[j,i] = sum(c[j:row:b])

#print chrm

chrm_new = np.zeros(b*col).reshape(b,col)

#median
"""
for i in range(b):
	for j in range(col):
		chrm_new[i,j] = np.median(chrm[i,j-3:j+3])
"""

chrm_new = chrm

chrm_new_norm = np.zeros(b*col).reshape(b,col)

#normalise
for i in range(col):
	chrm_new_norm[:,i] = chrm_new[:,i]/sum(chrm_new[:,i])
	#chrm_new_norm[i,j] = np.median(chrm_new[i,j-10:j+10])
#print chrm_new_norm


print "shape of Chroma is",shape(chrm_new_norm)
subplot(3,1,3)
title('Chroma (dB)')

#save chroma in a .npy file
chromaa = TemporaryFile()
np.save('chroma1.npy',chrm_new_norm)

im2 = imshow(chrm_new_norm,aspect='auto',interpolation='nearest',origin='lower')
#xlim(0,(len(sound_info)/11025.00))
yticks([1,3,5,7,9,11,13,15,17,19,21,23], ['A','A#','B','C','C#','D','D#','E','F','F#','G','G#'])
xticks(np.arange(0,col,172),[0,2,4,6,8,10,12,14])

#show()

#only for plotting
subplot(3,1,1)
title('Spectrogram (dB)')
Pxx, freqs, bins, im = spectrogram(sound_info, f, 6000, 128)
plot(Pxx,freqs)
xlim(0,(np.ceil(len(sound_info)/11025)))
show()


"""
Generate Template
"""

temp = generate_template()
xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23], 
	['C','C#','D','D#','E','F','F#','G','G#','A','A#','B','c','c#','d','d#','e','f','f#','g','g#','a','a#','b'])
yticks([0,1,2,3,4,5,6,7,8,9,10,11,12], ['A','A#','B','C','C#','D','D#','E','F','F#','G','G#'])
title('Template')
im = imshow(temp,aspect='auto',interpolation='nearest',origin='lower')
show()

print "template shape is",shape(temp)


"""
Fitness matrix computation by taking
dot product of Template and Chroma
"""

#Since b=24, we need to take mean of 2 rows
lol = zip(*chrm_new_norm[::-1])

print shape(lol)
lol = np.asarray(lol)
grand = lol.reshape(-1,2).mean(axis=1).reshape(lol.shape[0],-1)
#print shape(grand)

grand2 = zip(*grand)[::-1]
#print shape(grand2)

match = np.dot(temp.T,grand2)

match_norm = np.zeros(shape(match))

#normalize

for i in range(col):
	match_norm[:,i] = match[:,i]/sum(match[:,i])

print "matched array is",shape(match_norm)
title('Fitness Matrix')
im = imshow(match_norm,aspect='auto',interpolation='nearest',origin='lower')
yticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23], 
	['C','C#','D','D#','E','F','F#','G','G#','A','A#','B','c','c#','d','d#','e','f','f#','g','g#','a','a#','b'])
show()

#save tempogram in a .npy file
match22 = TemporaryFile()
np.save('tempogram1.npy',match_norm)

print 'end'



