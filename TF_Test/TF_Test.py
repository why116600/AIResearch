import tensorflow as tf
import numpy as np


def main():
	xdata=np.random.normal(0,10,(100000,1))
	ydata=np.square(xdata)
	xtest=np.random.normal()
	w1=tf.Variable(tf.random_normal([1,100],stddev=1,seed=1))
	w2=tf.Variable(tf.random_normal([100,100],stddev=1,seed=1))
	w3=tf.Variable(tf.random_normal([100,1],stddev=1,seed=1))
	b1=tf.Variable(tf.random_normal([1,100],stddev=1,seed=1))
	b2=tf.Variable(tf.random_normal([1,100],stddev=1,seed=1))
	b3=tf.Variable(tf.random_normal([1,1],stddev=1,seed=1))
	x=tf.placeholder(tf.float32,shape=(None,1),name='x-input')
	y_=tf.placeholder(tf.float32,shape=(None,1),name='y-input')

	h1=tf.nn.relu(tf.matmul(x,w1)+b1)
	h2=tf.nn.relu(tf.matmul(h1,w2)+b2)
	y=tf.matmul(h2,w3)+b3

	
	mse=tf.reduce_mean(tf.losses.mean_squared_error(y_,y))
	train_step=tf.train.AdamOptimizer(0.1).minimize(mse)

	with tf.Session() as sess:
		init_op=tf.initialize_all_variables()
		sess.run(init_op)
		print('first:')
		print(sess.run(w1))
		for i in range(0,xdata.shape[0],100):
			sess.run(train_step,feed_dict={x:xdata[i:i+100,:],y_:ydata[i:i+100,:]})
		print('now:')
		print(sess.run(w1))
		xtest=np.random.normal(0,10,(10,1))
		ytest=sess.run(y,feed_dict={x:xtest})
	print('X=',xtest)
	print('Y=',ytest)

if __name__=='__main__':
	main()
