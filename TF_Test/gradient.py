import tensorflow as tf
import numpy as np

def main():
	x=tf.Variable(0.0,dtype=tf.float32)
	y=(x-10.0)**2
	grad=tf.gradients(y,x)
	opti=tf.train.GradientDescentOptimizer(0.1)
	c_g=opti.compute_gradients(y)
	train_step=opti.apply_gradients(c_g)
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		g1=sess.run(grad)
		cg=sess.run(c_g)
		print(type(g1),':',g1)
		print(type(cg),':',cg)
		sess.run(train_step)
		print('x=',sess.run(x))
		g1=sess.run(grad)
		print(type(g1),':',g1)
		sess.run(train_step)


if __name__=='__main__':
	main()
