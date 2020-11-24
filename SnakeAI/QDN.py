from SnakeAI import SnakeEnv
from keras import models
from keras import layers
import random
import numpy as np
import os
import sys

class Network:
	def __init__(self,nInput):
		assert len(nInput)==2
		self.nInput=nInput
		self.model=models.Sequential()
		self.model.add(layers.Conv2D(10,(3,3),padding='same',activation='relu',input_shape=(nInput[0],nInput[1],1)))
		self.model.add(layers.Conv2D(10,(3,3),padding='same',activation='relu'))
		self.model.add(layers.Flatten())
		self.model.add(layers.Dense(1000,activation='relu'))
		self.model.add(layers.Dense(4))
		self.model.summary()
		self.model.compile(optimizer='rmsprop',loss='mse',metrics=['mae'])

	def LoadWeight(self,file):
		self.model.load_weights(file)

	def SaveWeight(self,file):
		self.model.save_weights(file)

	def Train(self,xdata,ydata):
		if len(xdata.shape)!=4:
			xdata=xdata.reshape((1,self.nInput[0],self.nInput[1],1))
		self.model.fit(xdata,ydata,epochs=10)

	def Calculate(self,xdata):
		if len(xdata.shape)==3:
			xdata=xdata.reshape((xdata.shape[0],self.nInput[0],self.nInput[1],1))
		else:
			xdata=xdata.reshape((1,self.nInput[0],self.nInput[1],1))
		return self.model.predict(xdata)

	def CopyTo(self,target):
		weights=self.model.get_weights()
		target.model.set_weights(weights)

class QDNAgent:
	ACTION=4
	def __init__(self,env):
		self.network=Network(env.shape)
		self.networkTarget=Network(env.shape)
		self.experiences=[]
		self.env=env

	def GetActionFromPolicy(self,state,epsilon=0.05):
		if random.random()>epsilon:
			qa=self.network.Calculate(state)
			return np.argmax(qa)
		else:
			return random.randint(0,self.ACTION-1)

	def Explore(self):
		S1=np.copy(self.env.state)
		A=self.GetActionFromPolicy(S1)
		S2,R,done,_=self.env.step(A)
		self.experiences.append((S1,A,R,done,np.copy(S2)))
		return R,done

	def Train(self,count=-1,gamma=0.9):
		pos=0
		states=[]
		qas=[]
		while pos!=count and len(self.experiences)>0:
			pos+=1
			choice=random.randint(0,len(self.experiences)-1)
			S1,A,R,done,S2=self.experiences[choice]
			qa=self.network.Calculate(np.array([S1,S2]))
			#qa[1]=self.network.Calculate(S2)
			maxA=np.argmax(qa[1].reshape(self.ACTION))
			tqa=self.networkTarget.Calculate(S2).reshape(self.ACTION)
			U=R
			if done==0:
				U+=gamma*tqa[maxA]
			qa[0,A]=U
			states.append(S1.reshape(self.env.shape[0],self.env.shape[1],1))
			qas.append(qa[0,:])
			#if float(R)<=1.0:
			#del self.experiences[choice]
		self.network.Train(np.array(states),np.array(qas))
		if len(self.experiences)>=1000000:
			self.experiences=[]

	def UpdateTargetNetwork(self):
		self.network.CopyTo(self.networkTarget)

	def Save(self,filepath):
		self.UpdateTargetNetwork()
		self.networkTarget.SaveWeight(filepath)

	def Load(self,filepath):
		self.networkTarget.LoadWeight(filepath)
		self.network.LoadWeight(filepath)

def GetTupleState(state):
	size=1
	for s in state.shape:
		size*=s
	return tuple(state.reshape(size))

def Main():
	if len(sys.argv)<3:
		return
	envSize=int(sys.argv[1])
	trainCount=int(sys.argv[2])
	roundLimit=envSize*envSize*envSize*envSize
	env=SnakeEnv((envSize,envSize))
	agent=QDNAgent(env)
	if len(sys.argv)>=4 and os.path.exists(sys.argv[3]):
		agent.Load(sys.argv[3])
	travel=0
	score=0.0
	maxScore=0.0
	print('first:')
	env.render()
	beforeStates=set([GetTupleState(env.state),])
	for i in range(trainCount):
		print('round ',i)
		R,done=agent.Explore()
		travel+=1
		score+=R
		print('step',travel)
		env.render()
		print('got score:',R)
		ts=GetTupleState(env.state)
		if ts in beforeStates:
			done=1
			print('repeat save state!')
		else:
			beforeStates.add(ts)
		if done!=0:
			travel=0
			if maxScore<score:
				maxScore=score
			print('final score:%f,max score:%f'%(score,maxScore))
			score=0.0
			env.reset()
			print('first')
			env.render()
			beforeStates=set([GetTupleState(env.state),])
		if (i%100)==99:
			agent.Train(count=100)
		if (i%1000)==100:
			if len(sys.argv)>=4:
				agent.Save(sys.argv[3])
			else:
				agent.UpdateTargetNetwork()
	print('Training complete!')
	if len(sys.argv)>=4:
		agent.Save(sys.argv[3])


if __name__=='__main__':
	Main()