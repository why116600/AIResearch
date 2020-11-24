import keras
from keras import models
from keras import layers
import gym
import numpy as np
import os

def build_network(input_s,output,activation=None,loss=keras.losses.MSE,learning_rate=0.01):
		model=models.Sequential()
		model.add(layers.Dense(16,activation='relu',input_shape=input_s))
		model.add(layers.Dense(16,activation='relu'))
		model.add(layers.Dense(output,activation=activation))
		model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss=loss)
		return model

class ACAgent:
	is_done=True
	act_file='./act.wdat'
	cri_file='./cri.wdat'
	def __init__(self,env,gamma=0.99,learning_rate=1.0):
		self.env=env
		self.gamma=gamma
		self.discount=1.0
		self.state_shape=env.observation_space.shape
		self.act_num=env.action_space.n
		self.actor_net=build_network(env.observation_space.shape,self.act_num,activation=keras.activations.softmax,loss=keras.losses.categorical_crossentropy)
		self.critic_net=build_network(env.observation_space.shape,self.act_num)
		if os.path.exists(self.act_file+'.index'):
			self.actor_net.load_weights(self.act_file)
		if os.path.exists(self.cri_file+'.index'):
			self.critic_net.load_weights(self.cri_file)

	def save_weight(self):
		self.actor_net.save_weights(self.act_file)
		self.critic_net.save_weights(self.cri_file)

	def decide(self,observation):
		probs = self.actor_net.predict(observation[np.newaxis])[0]
		action = np.random.choice(self.act_num, p=probs)
		return action

	def learn(self):
		if self.is_done:
			self.last_state=self.env.reset()
			self.discount=1.0
			self.last_action=self.decide(self.last_state)
		S=self.last_state
		A=self.last_action
		S1,R,done,_=self.env.step(A)
		A1=self.decide(S1)
		U=R
		if not done:
			U+=self.gamma*self.critic_net.predict(S1[np.newaxis])[0,A1]
		Q=self.critic_net.predict(S[np.newaxis])
		with tf.GradientTape() as tape:
			#tI=tf.convert_to_tensor(self.discount)
			tS=tf.convert_to_tensor(S[np.newaxis])
			q=Q[0,A]
			#tQ=self.critic_net(tS)[0,A]
			tPi=self.actor_net(tS)[0,A]
			y=-self.discount*q*tf.math.log(tf.clip_by_value(tPi,1e-6, 1.))
		grad_tensors = tape.gradient(y,self.actor_net.variables)
		self.actor_net.optimizer.apply_gradients(zip(grad_tensors, self.actor_net.variables))
		Q[0,A]=U
		self.critic_net.fit(S[np.newaxis],Q)
		self.discount*=self.gamma
		self.last_state=S1
		self.last_action=A1
		self.is_done=done
		return done


class A2CAgent:
	is_done=True
	act_file='./weight/act2.wdat'
	cri_file='./weight/cri2.wdat'
	target_score=200.0
	def __init__(self,env,gamma=0.99,learning_rate=0.01):
		self.env=env
		self.gamma=gamma
		self.discount=1.0
		self.state_shape=env.observation_space.shape
		self.act_num=env.action_space.n
		self.actor_net=build_network(env.observation_space.shape,self.act_num,activation=keras.activations.softmax,loss=keras.losses.categorical_crossentropy)
		self.critic_net=build_network(env.observation_space.shape,1)
		if os.path.exists(self.act_file):
			self.actor_net.load_weights(self.act_file)
		if os.path.exists(self.cri_file):
			self.critic_net.load_weights(self.cri_file)

	def save_weight(self):
		self.actor_net.save_weights(self.act_file)
		self.critic_net.save_weights(self.cri_file)

	def decide(self,observation):
		probs = self.actor_net.predict(observation[np.newaxis])[0]
		action = np.random.choice(self.act_num, p=probs)
		return action

	def learn(self):
		if self.is_done:
			self.last_state=self.env.reset()
			self.discount=1.0
			self.last_action=self.decide(self.last_state)
			self.score=0.0
		S=self.last_state
		A=self.last_action
		S1,R,done,_=self.env.step(A)
		print('choose action ',A)
		A1=self.decide(S1)
		if done:
			U=self.score-self.target_score
		else:
			U=R+self.gamma*self.critic_net.predict(S1[np.newaxis])[0,0]
			self.score+=1.0
		v=self.critic_net.predict(S[np.newaxis])[0,0]
		#with tf.GradientTape() as tape:
			#tS=tf.convert_to_tensor(S[np.newaxis])
			#tPi=self.actor_net(tS)[0,A]
			#y=-self.discount*(U-v)*tf.math.log(tf.clip_by_value(tPi,1e-6, 1.))
		#grad_tensors = tape.gradient(y,self.actor_net.variables)
		#self.actor_net.optimizer.apply_gradients(zip(grad_tensors, self.actor_net.variables))
		advantage=np.zeros((1,self.act_num))
		advantage[0,A]=U-v
		self.actor_net.fit(S[np.newaxis],advantage,epochs=1,verbose=0)
		self.critic_net.fit(S[np.newaxis],np.array([[U,],],dtype=np.float32),epochs=1,verbose=0)
		self.discount*=self.gamma
		self.last_state=S1
		self.last_action=A1
		self.is_done=done
		return done

def main():
	env=gym.make('CartPole-v1')
	agent=A2CAgent(env)
	round=0
	last=0
	step_num=0
	if os.path.exists('times.txt'):
		with open('times.txt','r') as fp:
			snum=fp.readline()
		step_num=int(snum)
	for i in range(999999):
		print('step%d'%(i,))
		done=agent.learn()
		if done:
			print('round%d last %d times'%(round,last))
			round+=1
			last=0
		else:
			last+=1
			if last>=agent.target_score:
				print('bingo!')
				agent.save_weight()
				with open('times2.txt','w') as fp:
					fp.write(str(step_num+i+1))
				return
			#env.render()
		if i%100==99:
			agent.save_weight()
			with open('times2.txt','w') as fp:
				fp.write(str(step_num+i+1))
	print('Game over!')


if __name__=='__main__':
	main()