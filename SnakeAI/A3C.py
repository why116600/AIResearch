#import tensorflow.compat.v2 as tf
import SnakeAI
import random
import numpy as np
import os
import sys
import multiprocessing

ENV_SIZE=8

def build_network(input_s,output,activation=None,loss='mse',learning_rate=0.01):
	import keras
	from keras import models
	from keras import layers
	#keras.backend.set_session()
	model=models.Sequential()
	model.add(layers.Conv2D(10,(3,3),padding='same',activation='relu',input_shape=(input_s[0],input_s[1],1)))
	model.add(layers.Conv2D(10,(3,3),padding='same',activation='relu'))
	model.add(layers.Flatten())
	model.add(layers.Dense(1000,activation='relu'))
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

def reshape_state(state):
	assert len(state.shape)==2
	return state.reshape(1,state.shape[0],state.shape[1],1)

class Trainer:
	TARGET_SCORE=ENV_SIZE**4
	def __init__(self,env,actor_weights=None,critic_weights=None,gamma=0.99,build_func=build_network):
		import keras
		self.env=env
		self.states=[]
		self.actions=[]
		self.rewards=[]
		self.gamma=gamma
		self.state_shape=env.shape
		self.act_num=env.action_num
		if type(actor_weights)==str and os.path.exists(actor_weights):
			self.actor_net=models.load_model(actor_weights)
			#self.actor_net.load_weights(actor_weights)
		else:
			self.actor_net=build_func(self.env.shape,self.act_num,activation=keras.activations.softmax,loss=keras.losses.categorical_crossentropy)
			if type(actor_weights)==list:
				set_network_weights(self.actor_net,actor_weights)
		if type(critic_weights)==str and os.path.exists(critic_weights):
			self.critic_net=models.load_model(critic_weights)
			#self.critic_net.load_weights(critic_weights)
		else:
			self.critic_net=build_func(self.env.shape,1)
			if type(critic_weights)==list:
				set_network_weights(self.critic_net,critic_weights)
		self.init_actor_weights=get_network_weights(self.actor_net)
		self.init_critic_weights=get_network_weights(self.critic_net)

	def decide(self,observation,epsilon=0.1):
		probs = self.actor_net.predict(reshape_state(observation))[0]
		action = np.random.choice(self.act_num, p=probs)
		return action

	def explore(self,limit,show_path=False):
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
		#if show_path:
			#print('first:')
			#print(S)
		for i in range(limit):
			S_,R_,done,_=self.env.step(A)
			self.states.append(S_)
			score+=R_
			if show_path:
				print('step',i)
				print('action:',A)
				#print(S_)
			if done:
				#if show_path:
					#print('get score:',score)
				#self.rewards.append(score-self.TARGET_SCORE)
				self.rewards.append(R_)
				self.is_done=True
				break
			self.rewards.append(R_)
			A=self.decide(S_)
			S=S_
			self.actions.append(A)
		return len(self.states)-1,score

	def pre_train(self):
		state_exp=np.zeros((0,)+self.state_shape+(1,))
		actor_exp=np.zeros((0,self.act_num))
		critic_exp=np.zeros((0,1))
		if self.is_done:
			U=0.0
		else:
			U=self.critic_net.predict(reshape_state(self.states[len(self.states)-1]))[0]
		for t in range(len(self.states)-2,-1,-1):
			S=self.states[t]
			U=self.gamma*U+self.rewards[t+1]
			v=self.critic_net.predict(reshape_state(S))[0]
			advantage=np.zeros((1,self.act_num))
			advantage[0,self.actions[t]]=U-v
			state_exp=np.vstack([state_exp,reshape_state(S)])
			actor_exp=np.vstack([actor_exp,advantage])
			if type(U)==np.ndarray:
				critic_exp=np.vstack([critic_exp,U.reshape((1,1))])
			else:
				critic_exp=np.vstack([critic_exp,np.array([[U,],],dtype=np.float32)])
		return state_exp,actor_exp,critic_exp

	def get_weights(self):
		return get_network_weights(self.actor_net),get_network_weights(self.critic_net)

	def save_to_file(self,actor_filen,criti_file):
		self.actor_net.save(actor_filen)
		self.critic_net.save(criti_file)

	def apply_experience(self,state_exp,actor_exp,critic_exp):
		if state_exp.shape[0]>0:
			self.actor_net.fit(state_exp,actor_exp,verbose=0,batch_size=state_exp.shape[0])
			self.critic_net.fit(state_exp,critic_exp,verbose=0,batch_size=state_exp.shape[0])



def queue_train_process(queue_h2c,queue_c2h):
	#print('start!')
	trainer=Trainer(SnakeAI.SnakeEnv((ENV_SIZE,ENV_SIZE),True))
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
	ACT_SUFFIX='_actor.h5'
	CRI_SUFFIX='_critic.h5'
	def __init__(self,env,filename,buidlfunc=build_network):
		self.env=env
		self.act_file=filename+self.ACT_SUFFIX
		self.critic_file=filename+self.CRI_SUFFIX
		self.mainTrainer=None

	def decide(self,observation):
		return self.mainTrainer.decide(observation)

	def start_training(self,processes_num):
		self.queue_h2c=[multiprocessing.Queue() for i in range(processes_num)]
		self.queue_c2h=[multiprocessing.Queue() for i in range(processes_num)]
		self.processes=[multiprocessing.Process(target=queue_train_process,args=(self.queue_h2c[i],self.queue_c2h[i])) for i in range(processes_num)]
		for proc in self.processes:
			proc.start()
		self.mainTrainer=Trainer(self.env,actor_weights=self.act_file,critic_weights=self.critic_file)

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

	def train(self,train_num):
		mission=train_num
		while mission>0:
			step_num,score=self.mainTrainer.explore(mission)
			mission-=step_num
			states,actors,critics=self.mainTrainer.pre_train()
			self.mainTrainer.apply_experience(states,actors,critics)

	def test(self):
		return self.mainTrainer.explore(ENV_SIZE**4,False)

	def save(self):
		self.mainTrainer.save_to_file(self.act_file,self.critic_file)

def main():
	env=SnakeAI.SnakeEnv((ENV_SIZE,ENV_SIZE),True)
	agent=A3CAgent(env,'a3cweights/snake')
	scores=[]
	max_score=-ENV_SIZE**4
	proc_num=4
	if len(sys.argv)>1:
		proc_num=int(sys.argv[1])
	agent.start_training(proc_num)
	for i in range(99999999):
		#agent.train(8*8*8*8)
		agent.train_round(ENV_SIZE**4)
		step_num,score=agent.test()
		scores.append(score)
		if score>max_score:
			max_score=score
		if len(scores)>=100:
			mean_score=np.mean(scores[-100:])
		else:
			mean_score=np.mean(scores)
		print('round %d insisted %d steps and got %d scores,max score:%f,average:%f'%(i,step_num,score,max_score,mean_score))
		if i%100==99:
			agent.save()

	agent.end_training()

if __name__=='__main__':
	main()
	