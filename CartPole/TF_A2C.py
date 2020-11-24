import tensorflow as tf
import numpy as np
import gym

class A2CNetwork:
	def __init__(self,state_shape,action_num,scope='default',learning_rate=0.01):
		#with tf.variable_scope(scope):
		self.action_num=action_num
		self.tf_state = tf.placeholder(dtype=tf.float32,shape=(None,state_shape[0]))
		weights1_a = tf.Variable(tf.random_normal([state_shape[0],32],stddev=.1,seed=1))
		weights2_a = tf.Variable(tf.random_normal([32,32],stddev=.1,seed=1))
		weights1_c = tf.Variable(tf.random_normal([state_shape[0],32],stddev=.1,seed=1))
		weights2_c = tf.Variable(tf.random_normal([32,32],stddev=.1,seed=1))
		self.weights3_a = tf.Variable(tf.random_normal([32,action_num],stddev=.1,seed=1))
		weights3_c = tf.Variable(tf.random_normal([32,1],stddev=.1,seed=1))

		
		bias1_a = tf.Variable(tf.constant(0.1,dtype=tf.float32,shape=(1,32)))
		bias2_a = tf.Variable(tf.constant(0.1,dtype=tf.float32,shape=(1,32)))
		bias1_c = tf.Variable(tf.constant(0.1,dtype=tf.float32,shape=(1,32)))
		bias2_c = tf.Variable(tf.constant(0.1,dtype=tf.float32,shape=(1,32)))
		self.bias3_a = tf.Variable(tf.constant(0.1,dtype=tf.float32,shape=(1,action_num)))
		bias3_c = tf.Variable(tf.constant(0.1,dtype=tf.float32,shape=(1,1)))

		h1_a = tf.nn.relu(tf.matmul(self.tf_state,weights1_a) + bias1_a)
		h2_a = tf.nn.relu(tf.matmul(h1_a,weights2_a) + bias2_a)

		h1_c = tf.nn.relu(tf.matmul(self.tf_state,weights1_c) + bias1_c)
		h2_c = tf.nn.relu(tf.matmul(h1_c,weights2_c) + bias2_c)

		self.tf_act=tf.matmul(h2_a,self.weights3_a) + self.bias3_a
		self.tf_action = tf.nn.softmax(self.tf_act)
		self.tf_action_=tf.placeholder(dtype=tf.float32,shape=(None,action_num))
		#self.tf_index=tf.placeholder(dtype=tf.int32,shape=(None,2))
		#self.tf_pi_ = tf.placeholder(dtype=tf.float32,shape=(None,1))

		#cross_entropy = tf.reduce_mean(-self.tf_pi_*tf.log(tf.gather_nd(self.tf_action,self.tf_index)))
		cross_entropy=-tf.reduce_sum(self.tf_action_*tf.log(self.tf_action),reduction_indices=[1,])
		#self.train_actor=[tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropies[i]) for i in range(action_num)]
		#self.train_actor=tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
		self.train_actor = tf.train.AdamOptimizer(learning_rate/10.0).minimize(cross_entropy)

		self.tf_value = tf.matmul(h2_c,weights3_c) + bias3_c
		self.tf_value_ = tf.placeholder(dtype=tf.float32,shape=(None,1))

		mse = tf.reduce_mean(tf.losses.mean_squared_error(self.tf_value_,self.tf_value))
		#self.train_critic = tf.train.GradientDescentOptimizer(learning_rate).minimize(mse)
		self.train_critic = tf.train.AdamOptimizer(learning_rate).minimize(mse)

		init_op = tf.global_variables_initializer()

		self.session=tf.Session()
		self.session.run(init_op)


	def __del__(self):
		self.session.close()

	def to_train_actor(self,state,action,pi):
		if len(state.shape)==1:
			state=state[np.newaxis]
		if type(pi)!=np.ndarray:
			pi_=pi
			pi=np.zeros((1,self.action_num))
			pi[0,action]=pi_
		#self.session.run(self.train_actor,feed_dict={self.tf_state:state,self.tf_index:np.array([[0,action],]),self.tf_action_:pi})
		self.session.run(self.train_actor,feed_dict={self.tf_state:state,self.tf_action_:pi})
		w3=self.session.run(self.weights3_a)
		#print('sum(w3[0])=',np.sum(w3[:,0]))
		#print('sum(w3[1])=',np.sum(w3[:,1]))
		#print('b3=',self.session.run(self.bias3_a))

	def to_train_critic(self,state,U):
		if len(state.shape)==1:
			state=state[np.newaxis]
		if type(U)!=np.ndarray:
			U=np.ones((1,1))*U
		#a=self.action_probability(state)
		self.session.run(self.train_critic,feed_dict={self.tf_state:state,self.tf_value_:U})

	def evaluate(self,state):
		if len(state.shape)==1:
			state=state[np.newaxis]
		return self.session.run(self.tf_value,feed_dict={self.tf_state:state})

	def action_probability(self,state):
		if len(state.shape)==1:
			state=state[np.newaxis]
		pi=self.session.run(self.tf_act,feed_dict={self.tf_state:state})
		return self.session.run(self.tf_action,feed_dict={self.tf_state:state})

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
			self.last_action=self.decide(self.last_state)
			self.score=0.0
		S=self.last_state
		A=self.last_action
		S1,R,done,_=self.env.step(A)
		#print('choose action ',A)
		A1=self.decide(S1)
		if done:
			U=self.score-200.0
		else:
			U=R+self.gamma*self.network.evaluate(S1)[0,0]
			self.score+=1.0
		v=self.network.evaluate(S)[0,0]
		#print('state value:',v)
		#advantage=np.zeros((1,self.action_num))
		#advantage[0,A]=U-v
		self.network.to_train_actor(S,A,self.discount*(U-v))
		self.network.to_train_critic(S,U)
		self.discount*=self.gamma
		self.last_state=S1
		self.last_action=A1
		self.is_done=done
		self.discount*=self.gamma
		return done

	
def main():
	env=gym.make('CartPole-v1')
	agent=A2CAgent(env)
	round=0
	last=0
	step_num=0
	for i in range(999999):
		#print('step%d'%(i,))
		done=agent.learn()
		if done:
			print('round%d last %d times'%(round,last))
			round+=1
			last=0
		else:
			last+=1
			if last>=200:
				print('bingo!')
				return
	print('Game over!')


if __name__=='__main__':
	main()