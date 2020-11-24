import tensorflow.compat.v2 as tf
from tensorflow import keras

def main():
	model=keras.models.Sequential()
	model.add(keras.layers.Dense(1,input_shape=(1,1)))
	model.summary()
	model.compile(optimizer=tf.optimizers.Adam(1.0),loss='mse',metrics=['mae'])
	a=tf.Variable(tf.constant(-1.0))
	b=tf.Variable(tf.constant(20.0))
	c=tf.Variable(tf.constant(100.0))
	print('x0=',model.variables)
	for i in range(100):
		with tf.GradientTape() as tape:
			#x=tf.Variable(tf.constant(0.0))
			x=model(tf.convert_to_tensor([[0.0,],],dtype=tf.float32))[0]
			y=a*x**2+b*x+c
		dx=tape.gradient(y,model.variables)
		print('dx=',[t.numpy() for t in dx])
		print('y=',y.numpy())
		ddx=[t*tf.constant([-1.0,]) for t in dx]
		model.optimizer.apply_gradients(zip(ddx, model.variables))
		print('x=',[t.numpy() for t in model.variables])
		

if __name__=='__main__':
	main()
