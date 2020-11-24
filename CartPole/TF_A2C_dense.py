import tensorflow as tf
import numpy as np
import gym

class A2CNetwork:
	def __init__(self,state_shape,action_num,scope='default',gamma=0.9,learning_rate=0.001):
		#with tf.variable_scope(scope):
		self.tf_state = tf.placeholder(dtype=tf.float32,shape=(1,state_shape[0]))
		h1_a=tf.layers.dense(self.tf_state,20,
					   activation=tf.nn.relu,
					   kernel_initializer=tf.random_normal_initializer(0., .1),
					   bias_initializer=tf.constant_initializer(0.1))
		h2_a=tf.layers.dense(h1_a,20,
					   activation=tf.nn.relu,
					   kernel_initializer=tf.random_normal_initializer(0., .1),
					   bias_initializer=tf.constant_initializer(0.1))

		self.tf_pi=tf.layers.dense(h2_a,action_num,
					   activation=tf.nn.softmax,
					   kernel_initializer=tf.random_normal_initializer(0., .1),
					   bias_initializer=tf.constant_initializer(0.1))

		h1_c=tf.layers.dense(self.tf_state,20,
					   activation=tf.nn.relu,
					   kernel_initializer=tf.random_normal_initializer(0., .1),
					   bias_initializer=tf.constant_initializer(0.1))
		h2_c=tf.layers.dense(h1_c,20,
					   activation=tf.nn.relu,
					   kernel_initializer=tf.random_normal_initializer(0., .1),
					   bias_initializer=tf.constant_initializer(0.1))

		self.tf_value=tf.layers.dense(h2_c,1,
					   activation=None,
					   kernel_initializer=tf.random_normal_initializer(0., .1),
					   bias_initializer=tf.constant_initializer(0.1))

		self.tf_action=tf.placeholder(dtype=tf.int32)
		self.tf_reward=tf.placeholder(dtype=tf.float32)
		self.tf_discount=tf.placeholder(dtype=tf.float32)

		self.tf_next_value=tf.placeholder(dtype=tf.float32)
		U=self.tf_reward+tf.constant(gamma)*self.tf_next_value

		mse = tf.square(U-self.tf_value)#tf.reduce_mean(tf.losses.mean_squared_error(U,self.tf_value))
		self.train_critic = tf.train.GradientDescentOptimizer(learning_rate).minimize(mse)
		#self.train_critic = tf.train.AdamOptimizer(learning_rate).minimize(mse)

		#cross_entropy = tf.reduce_mean(-(U-self.tf_value)*tf.log(tf.gather_nd(self.tf_pi,self.tf_action)))
		#cross_entropy = tf.reduce_sum(self.tf_discount*(U-self.tf_value)*tf.log(self.tf_pi[0,self.tf_action]))
		#cross_entropy=tf.reduce_mean(-tf.reduce_sum(self.tf_action_*tf.log(self.tf_action),reduction_indices=[1,]))
		cross_entropy = tf.reduce_sum(self.tf_discount*(U-self.tf_value)*tf.log(self.tf_pi[0,self.tf_action]))
		#self.train_actor=[tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropies[i]) for i in range(action_num)]
		self.train_actor=tf.train.GradientDescentOptimizer(learning_rate).minimize(-cross_entropy)
		#self.train_actor = tf.train.AdamOptimizer(learning_rate/10.0).minimize(-cross_entropy)
		self.train_step=tf.train.AdamOptimizer(learning_rate).minimize(mse-cross_entropy)


		init_op = tf.global_variables_initializer()

		self.session=tf.Session()
		self.session.run(init_op)


	def __del__(self):
		self.session.close()


	def train(self,state,action,reward,next_state,discount=1.0):
		if len(state.shape)==1:
			state=state[np.newaxis]
		if len(next_state.shape)==1:
			next_state=next_state[np.newaxis]
		new_value=self.evaluate(next_state)
		fd={self.tf_state:state,self.tf_action:action,self.tf_reward:reward,self.tf_next_value:new_value,self.tf_discount:discount}
		self.session.run(self.train_critic,feed_dict=fd)
		self.session.run(self.train_actor,feed_dict=fd)
		#self.session.run(self.train_step,feed_dict=fd)

	def evaluate(self,state):
		if len(state.shape)==1:
			state=state[np.newaxis]
		return self.session.run(self.tf_value,feed_dict={self.tf_state:state})

	def action_probability(self,state):
		if len(state.shape)==1:
			state=state[np.newaxis]
		return self.session.run(self.tf_pi,feed_dict={self.tf_state:state})

class A2CAgent:
	def __init__(self,env,gamma=0.99):
		self.discount=1.0
		self.env=env
		self.gamma=gamma
		self.action_num=env.action_space.n
		self.network=A2CNetwork(env.observation_space.shape,env.action_space.n)
		self.is_done=True

	def decide(self,state):
		probs=self.network.action_probability(state)
		return np.random.choice(self.action_num,p=probs[0])

	def learn(self):
		if self.is_done:
			self.last_state=self.env.reset()
			self.discount=1.0
			self.score=0.0
		S=self.last_state
		A=self.decide(S)
		S1,R,done,_=self.env.step(A)
		#print('choose action ',A)
		#if done:
			#R=-20.0
		self.network.train(S,A,R,S1)
		self.discount*=self.gamma
		self.last_state=S1
		self.is_done=done
		self.discount*=self.gamma
		return done

	
def main():
	env=gym.make('CartPole-v1')
	env.seed(1)  # reproducible
	env = env.unwrapped
	agent=A2CAgent(env)
	round=0
	last=0
	step_num=0
	for i in range(999999):
		done=agent.learn()
		if done:
			print('=======round%d last %d times========'%(round,last))
			round+=1
			last=0
		else:
			last+=1
			if last>=200:
				print('bingo!')
				print('used step%d'%(i,))
				return
	print('Game over!')


if __name__=='__main__':
	main()