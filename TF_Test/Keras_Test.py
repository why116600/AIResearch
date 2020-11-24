import tensorflow as tf
from keras import models
from keras import layers
import numpy as np


def main1():
	xdata=np.random.normal(0,10,100000)
	ydata=np.square(xdata)
	xtest=np.random.normal()
	model=models.Sequential()
	model.add(layers.Dense(100,activation='relu',input_shape=(1,)))
	model.add(layers.Dense(100,activation='relu'))
	model.add(layers.Dense(1))
	model.compile(optimizer='rmsprop',loss='mse',metrics=['mae'])
	for i in range(10):
		model.fit(xdata,ydata,batch_size=100)
	#model.fit(xdata,ydata,epochs=20,batch_size=100)
	xtest=np.random.normal(0,10,10)
	ytest=model.predict(xtest)
	print('X=',xtest)
	print('Y=',ytest)

def main():
	xdata=np.random.normal(0,10,100000)
	ydata=np.square(xdata)
	xtest=np.random.normal()
	model=tf.keras.models.Sequential()
	model.add(tf.keras.layers.Dense(100,activation='relu',input_shape=(1,)))
	model.add(tf.keras.layers.Dense(100,activation='relu'))
	model.add(tf.keras.layers.Dense(1))
	model.compile(optimizer='rmsprop',loss='mse',metrics=['mae'])
	for i in range(10):
		model.fit(xdata,ydata,batch_size=100)
	#model.fit(xdata,ydata,epochs=20,batch_size=100)
	xtest=np.random.normal(0,10,10)
	ytest=model.predict(xtest)
	print('X=',xtest)
	print('Y=',ytest)

if __name__=='__main__':
	main()
