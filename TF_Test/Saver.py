import tensorflow as tf
import numpy as np
import os

class Model:
	def __init__(self,name):
		with tf.variable_scope(name):
			self.x=tf.placeholder(dtype=tf.float32,shape=None,name='x')
			self.k=tf.Variable(tf.constant(1.0,dtype=tf.float32,shape=None,name='k'))
			self.b=tf.Variable(tf.constant(1.0,dtype=tf.float32,shape=None,name='y'))
			self.y=self.k*self.x+self.b
			self.next_k=tf.placeholder(dtype=tf.float32,shape=None,name='k_')
			self.next_b=tf.placeholder(dtype=tf.float32,shape=None,name='b_')
			self.set_k=tf.assign(self.k,self.next_k)
			self.set_b=tf.assign(self.b,self.next_b)
			self.init=tf.initialize_all_variables()
			self.saver=tf.train.Saver()

		self.sess=tf.Session()
		self.sess.run(self.init)

	def __del__(self):
		self.sess.close()

	def calculate(self,x):
		return self.sess.run(self.y,feed_dict={self.x:x})

	def set_k_value(self,k):
		return self.sess.run(self.set_k,feed_dict={self.next_k:k})

	def set_b_value(self,b):
		return self.sess.run(self.set_b,feed_dict={self.next_b:b})

	def save(self,path):
		self.saver.save(self.sess,path)

	def load(self,path):
		self.saver.restore(self.sess,path)
		#self.sess.run(self.saver)

PATH1='a.ckpt'
PATH2='b.ckpt'

if __name__=='__main__':
	model1=Model('a')
	model2=Model('b')
	if os.path.exists(PATH1+'.index'):
		model1.load(PATH1)
	if os.path.exists(PATH2+'.index'):
		model2.load(PATH2)
	print('first:')
	print('model1 value:',model1.calculate(1))
	print('model2 value:',model2.calculate(1))
	k=float(input('input k1:'))
	b=float(input('input b1:'))
	model1.set_k_value(k)
	model1.set_b_value(b)
	k=float(input('input k2:'))
	b=float(input('input b2:'))
	model2.set_k_value(k)
	model2.set_b_value(b)
	print('now:')
	print('model1 value:',model1.calculate(1))
	print('model2 value:',model2.calculate(1))
	model1.save(PATH1)
	model2.save(PATH2)
