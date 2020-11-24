#import tensorflow.compat.v2 as tf
import keras
from keras import models
from keras import layers
import gym
import random
import numpy as np
import os
import sys
import multiprocessing


def build_network(input_s,output,activation=None,loss=keras.losses.MSE,learning_rate=0.01):
		model=models.Sequential()
		model.add(layers.Dense(16,activation=keras.activations.relu,input_shape=input_s))
		model.add(layers.Dense(16,activation=keras.activations.relu))
		model.add(layers.Dense(output,activation=activation))
		model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss=loss)
		return model

def set_network_weights(model,all_weights):
	assert len(model.layers)==len(all_weights)
	for i,weights in enumerate(all_weights):
		model.layers[i].set_weights(weights)

def get_network_weights(model):
	results=[]
	for layer in model.layers:
		weights=layer.get_weights()
		results.append(weights)
	return results

def update_network_weights(model,start_weights,end_weights):
	assert len(model.layers)==len(start_weights) and len(model.layers)==len(end_weights)
	for i,layer in enumerate(model.layers):
		old_weights=layer.get_weights()
		delta_weights=[end_weights[i][j]-start_weights[i][j] for j in range(len(start_weights[i]))]
		layer.set_weights([old_weights[j]+delta_weights[j] for j in range((len(old_weights)))])


class Trainer:
	TARGET_SCORE=200.0
	def __init__(self,env,actor_weights=None,critic_weights=None,gamma=0.99,build_func=build_network):
		self.env=env
		self.states=[]
		self.actions=[]
		self.rewards=[]
		self.gamma=gamma
		self.state_shape=env.observation_space.shape
		self.act_num=env.action_space.n
		self.actor_net=build_func(self.env.observation_space.shape,self.act_num,activation=keras.activations.softmax,loss=keras.losses.categorical_crossentropy)
		self.critic_net=build_func(self.env.observation_space.shape,1)
		if type(actor_weights)==str and os.path.exists(actor_weights):
			self.actor_net.load_weights(actor_weights)
		elif type(actor_weights)==list:
			set_network_weights(self.actor_net,actor_weights)
		if type(critic_weights)==str and os.path.exists(critic_weights):
			self.critic_net.load_weights(critic_weights)
		elif type(critic_weights)==list:
			set_network_weights(self.critic_net,critic_weights)
		self.init_actor_weights=get_network_weights(self.actor_net)
		self.init_critic_weights=get_network_weights(self.critic_net)

	def decide(self,observation):
		probs = self.actor_net.predict(observation[np.newaxis])[0]
		action = np.random.choice(self.act_num, p=probs)
		return action

	def explore(self,limit):
		S=self.env.reset()
		A=self.decide(S)
		del self.states
		del self.actions
		del self.rewards
		self.states=[S,]
		self.actions=[A,]
		self.rewards=[0.0,]
		self.is_done=False
		score=0.0
		for i in range(limit):
			S_,R_,done,_=self.env.step(A)
			#if self.train_num<=0:
				#print('choose action ',A)
			self.states.append(S_)
			score+=R_
			if done:
				#if self.train_num<=0:
					#print('get score:',score)
				self.rewards.append(score-self.TARGET_SCORE)
				self.is_done=True
				break
			self.rewards.append(R_)
			A=self.decide(S_)
			S=S_
			self.actions.append(A)
		return len(self.states)-1

	def pre_train(self):
		state_exp=np.zeros((0,)+self.state_shape)
		actor_exp=np.zeros((0,self.act_num))
		critic_exp=np.zeros((0,1))
		if self.is_done:
			U=0.0
		else:
			U=self.critic_net.predict(self.states[len(self.states)-1][np.newaxis])[0]
		for t in range(len(self.states)-2,-1,-1):
			S=self.states[t]
			U=self.gamma*U+self.rewards[t+1]
			v=self.critic_net.predict(S[np.newaxis])[0]
			advantage=np.zeros((1,self.act_num))
			advantage[0,self.actions[t]]=U-v
			state_exp=np.vstack([state_exp,S[np.newaxis]])
			actor_exp=np.vstack([actor_exp,advantage])
			if type(U)==np.ndarray:
				critic_exp=np.vstack([critic_exp,U.reshape((1,1))])
			else:
				critic_exp=np.vstack([critic_exp,np.array([[U,],],dtype=np.float32)])
		return state_exp,actor_exp,critic_exp

	def get_weights(self):
		return get_network_weights(self.actor_net),get_network_weights(self.critic_net)

	def save_to_file(self,actor_filen,criti_file):
		self.actor_net.save_weights(actor_filen)
		self.critic_net.save_weights(criti_file)

	def apply_experience(self,state_exp,actor_exp,critic_exp):
		if state_exp.shape[0]>0:
			self.actor_net.fit(state_exp,actor_exp,verbose=0,batch_size=state_exp.shape[0])
			self.critic_net.fit(state_exp,critic_exp,verbose=0,batch_size=state_exp.shape[0])


def train_process(ator_weights,critic_weights,train_num):
	trainer=Trainer(gym.make('CartPole-v1'),ator_weights,critic_weights)
	trainer.explore(train_num)
	return trainer.pre_train()

def queue_train_process(queue_h2c,queue_c2h):
	#print('start!')
	trainer=Trainer(gym.make('CartPole-v1'))
	while True:
		message=queue_h2c.get()
		#print('get message:',message)
		if type(message)==str:
			break
		if type(message)==tuple and len(message)==3:
			actor_weights,critic_weights,train_num=message
			set_network_weights(trainer.actor_net,actor_weights)
			set_network_weights(trainer.critic_net,critic_weights)
			trainer.explore(train_num)
			queue_c2h.put(trainer.pre_train())
	
class A3CAgent:
	ACT_SUFFIX='_actor'
	CRI_SUFFIX='_critic'
	def __init__(self,env,filename,buidlfunc=build_network):
		self.act_file=filename+self.ACT_SUFFIX
		self.critic_file=filename+self.CRI_SUFFIX
		self.mainTrainer=Trainer(env,actor_weights=self.act_file,critic_weights=self.critic_file)

	def decide(self,observation):
		return self.mainTrainer.decide(observation)

	def start_training(self,processes_num):
		actor_weights,critic_weights=self.mainTrainer.get_weights()
		self.queue_h2c=[multiprocessing.Queue() for i in range(processes_num)]
		self.queue_c2h=[multiprocessing.Queue() for i in range(processes_num)]
		self.processes=[multiprocessing.Process(target=queue_train_process,args=(self.queue_h2c[i],self.queue_c2h[i])) for i in range(processes_num)]
		for proc in self.processes:
			proc.start()

	def train_round(self,train_num):
		actor_weights,critic_weights=self.mainTrainer.get_weights()
		for queue in self.queue_h2c:
			queue.put((actor_weights,critic_weights,train_num),False)
		for queue in self.queue_c2h:
			states,actors,critics=queue.get()
			self.mainTrainer.apply_experience(states,actors,critics)

	def end_training(self):
		for queue in self.queue_h2c:
			queue.put("exit",False)
		for proc in self.processes:
			proc.join()
			proc.terminate()

	def train(self,processes_num,train_num):
		if processes_num<=1:
			mission=train_num
			while mission>0:
				mission-=self.mainTrainer.explore(mission)
				states,actors,critics=self.mainTrainer.pre_train()
				#print(mission,' missions left')
				self.mainTrainer.apply_experience(states,actors,critics)
				#print('Merging experience complete!')
			#print('one training complete!')
			return
		actor_weights,critic_weights=self.mainTrainer.get_weights()
		pool=multiprocessing.Pool(processes_num)
		results=[pool.apply_async(train_process,(actor_weights,critic_weights,train_num)) for i in range(processes_num)]
		pool.close()
		pool.join()
		for res in results:
			states,actors,critics=res.get()
			self.mainTrainer.apply_experience(states,actors,critics)

	def test(self):
		return self.mainTrainer.explore(500)

	def save(self,filename):
		self.mainTrainer.save_to_file(self.act_file,self.critic_file)

def main():
	env=gym.make('CartPole-v1')
	agent=A3CAgent(env,'a3cweight/cartpole')
	agent.start_training(4)
	for i in range(1000):
		agent.train_round(400)
		score=agent.test()
		print('round %d got %d scores'%(i,score))
		agent.save('cartpole')
		if score>=200:
			print('bingo!')
			break
	agent.end_training()

if __name__=='__main__':
	main()
	