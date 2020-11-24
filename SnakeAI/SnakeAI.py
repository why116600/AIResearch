import numpy as np
import random
import sys
import os
from keras import models
from keras import layers

class Network:
	def __init__(self,nInput):
		self.nInput=nInput
		self.model=models.Sequential()
		self.model.add(layers.Dense(100,activation='relu',input_shape=(nInput,)))
		self.model.add(layers.Dense(100,activation='relu'))
		self.model.add(layers.Dense(1))
		self.model.compile(optimizer='rmsprop',loss='mse',metrics=['mae'])

	def LoadWeight(self,file):
		self.model.load_weights(file)

	def SaveWeight(self,file):
		self.model.save_weights(file)

	def Train(self,xdata,ydata):
		self.model.fit(xdata,ydata,epochs=5)

	def Calculate(self,xdata):
		if len(xdata.shape)==1:
			xdata=xdata.reshape((1,self.nInput))
		return self.model.predict(xdata)

EMPTY_STATE=-1
FOOD_STATE=-2

def snake_state(s):
	if s>=0:
		return 1
	return 0

SnakeState=np.vectorize(snake_state)

DirectionMap={0:np.array([0,-1]),1:np.array([-1,0]),2:np.array([1,0]),3:np.array([0,1])}#0-up,1-left,2-right,3-down

class SnakeEnv:
	action_num=4
	def __init__(self,shape,simple_reward=False):
		assert len(shape)==2
		if simple_reward:
			self.food_score=1.0
			self.forward_score=0.0
			self.dead_score=-shape[0]*shape[1]
		else:
			self.food_score=shape[0]*shape[1]+1.0
			self.forward_score=1.0
			self.dead_score=0.0
		self.shape=shape
		self.grid_size=shape[0]*shape[1]
		self.reset()

	def IndexToPos(self,index):
		if index<0:
			return (-1,-1)
		return (index%self.shape[1],index//self.shape[1])
	
	def reset(self):
		start=random.randint(0,self.grid_size-1)
		food=random.randint(0,self.grid_size-1)
		while food==start:
			food=random.randint(0,self.grid_size-1)
		self.food=food
		self.direction=random.randint(0,3)
		self.state=np.ones(self.shape,dtype=int)*EMPTY_STATE
		fp=self.IndexToPos(food)
		self.state[fp]=FOOD_STATE
		self.state[self.IndexToPos(start)]=0
		self.snake=[self.IndexToPos(start),]
		return self.state

	def RefreshState(self,set_food=False,start=0,kick=-1):
		self.state=np.ones(self.shape,dtype=int)*EMPTY_STATE
		for i,pos in enumerate(self.snake):
			if i>=start and i!=kick:
				self.state[pos]=i
		if set_food:
			food=random.randint(0,self.grid_size-1)
			while self.state[self.IndexToPos(food)]>=0:
				food=random.randint(0,self.grid_size-1)
			self.food=food
		self.state[self.IndexToPos(self.food)]=FOOD_STATE

	def step(self,action):#return state,reward,complete,remark
		if action not in DirectionMap.keys():
			return self.state,0,0,{}
		head=self.snake[0]
		next=tuple(np.array(head)+DirectionMap[action])
		#going outside
		if next[0]<0 or next[0]>=self.shape[0] or next[1]<0 or next[1]>=self.shape[1]:
			self.snake=[self.snake[0],]+self.snake[0:-1]
			self.RefreshState(start=1)
			return self.state,self.dead_score,1,{}
		s_next=self.state[next]
		#eat itself
		if (s_next>=0 and s_next!=(len(self.snake)-1)) or s_next==1:
			self.snake=[next,]+self.snake[0:-1]
			self.RefreshState(kick=s_next)
			return self.state,self.dead_score,1,{}
		#eat food
		if s_next==FOOD_STATE:
			self.snake=[next,]+self.snake
			if len(self.snake)>=self.grid_size:
				self.RefreshState()
				return self.state,self.food_score,1,{}
			self.RefreshState(True)
			return self.state,self.food_score,0,{}
		#just move forward
		self.snake=[next,]+self.snake[0:-1]
		self.RefreshState()
		forward=np.sum(np.abs(np.array(head)-np.array(self.IndexToPos(self.food))))-\
			np.sum(np.abs(np.array(next)-np.array(self.IndexToPos(self.food))))
		return self.state,forward*self.forward_score,0,{}#0,0,{}#

	def reward(self,s):
		return np.sum(SnakeState(s))

	def render(self):
		print(self.state)

class SnakeAgent:
	def __init__(self,env):
		self.env=env
		self.network=Network(env.grid_size+1)

	def ActionFromPolicy(self,state,IbuSloan=0.1):
		values=[self.network.Calculate(np.append(state.reshape(1,self.env.grid_size),np.array([a,]))) for a in range(4)]
		nAction=len(DirectionMap.keys())
		policy=np.ones(nAction)*IbuSloan/nAction
		maxA=np.argmax(values)
		policy[maxA]+=1-IbuSloan
		rmd=random.random()
		frmd=rmd
		A=0
		while rmd>=0.0:
			rmd-=policy[A]
			if rmd<=0.0:
				break
			A+=1
		return A
		
	def Train(self,gamma=0.9,travel_limit=-1):
		travel=0
		rewardG=0.0
		self.env.reset()
		# action trace
		states=[self.env.state,]
		print('first:')
		self.env.render()
		rewards=[0.0,]
		actions=[]
		finish=0
		past_states=set()
		ts=tuple(s for s in self.env.state.reshape(self.env.grid_size))
		past_states.add(ts)
		while finish==0 and travel!=travel_limit:
			A=self.ActionFromPolicy(states[-1])
			S,R,finish,remark=self.env.step(A)
			rewards.append(R)
			actions.append(A)
			states.append(S)
			travel+=1
			print('step',travel)
			self.env.render()
			if R>0:
				print('got %f point'%(R,))
			ts=tuple(s for s in S.reshape(self.env.grid_size))
			if ts in past_states:
				print('State recursed!')
				break
			past_states.add(ts)
		xdata=[]
		ydata=[]
		for t in range(len(states)-2,-1,-1):
			rewardG=gamma*rewardG+rewards[t+1]
			xdata.append(np.append(states[t].reshape(1,self.env.grid_size),np.array([actions[t],])))
			ydata.append(np.array([rewardG,]))
		self.network.Train(np.vstack(xdata),np.vstack(ydata))
		return np.sum(rewards)

def main():
	if len(sys.argv)<3:
		return
	maxScore=0.0
	allScore=0.0
	envSize=int(sys.argv[1])
	trainCount=int(sys.argv[2])
	env=SnakeEnv((envSize,envSize))
	agent=SnakeAgent(env)
	if len(sys.argv)>=4 and os.path.exists(sys.argv[3]):
		agent.network.LoadWeight(sys.argv[3])
	for i in range(trainCount):
		print('Round ',i)
		score=agent.Train(travel_limit=envSize*envSize*envSize*envSize)
		if score>maxScore:
			maxScore=score
		allScore+=score
		print('score:%f,max score:%f,average score:%f'%(score,maxScore,allScore/(i+1)))
		if len(sys.argv)>=4 and (i%1000)==999:
			agent.network.SaveWeight(sys.argv[3])
	print('training complete!')
	if len(sys.argv)>=4:
		agent.network.SaveWeight(sys.argv[3])
	

if __name__=='__main__':
	main()
