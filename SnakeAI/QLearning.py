import numpy as np
import random
import sys
import os
from SnakeAI import SnakeEnv
from SnakeAI import Network

class DoubleQLearningAgent:
	def __init__(self,env,action_n=4,gamma=0.9,learning_rate=0.1):
		self.env=env
		self.gamma=gamma
		self.learning_rate=learning_rate
		self.q0={}
		self.q1={}
		self.action_n=action_n

	def Save(self,filename):
		with open(filename,'w') as fp:
			fp.write('%d,%d,%d\n'%(self.env.grid_size,len(self.q0.keys()),len(self.q1.keys())))
			for state,action in self.q0.keys():
				for i in range(len(state)):
					fp.write('%d,'%state[i])
				fp.write('%d,'%action)
				fp.write(str(self.q0[(state,action)]))
				fp.write('\n')
			for state,action in self.q1.keys():
				for i in range(len(state)):
					fp.write('%d,'%state[i])
				fp.write('%d,'%action)
				fp.write(str(self.q1[(state,action)]))
				fp.write('\n')

	def Open(self,filename):
		with open(filename,'r') as fp:
			head=fp.readline()
			gridsize,nq0,nq1=[int(s) for s in head.split(',')]
			self.q0={}
			self.q1={}
			for i in range(nq0):
				line=fp.readline()
				data=line.split(',')
				if len(data)<=2:
					continue
				state=tuple([int(s) for s in data[:-2]])
				action=int(data[-2])
				value=float(data[-1])
				self.q0[(state,action)]=value
			for i in range(nq1):
				line=fp.readline()
				data=line.split(',')
				if len(data)<=2:
					continue
				state=tuple([int(s) for s in data[:-2]])
				action=int(data[-2])
				value=float(data[-1])
				self.q1[(state,action)]=value


	def GetQ0Value(self,state,action):
		value=0.0
		if state in self.q0.keys():
			value=self.q0[(state,action)]
		return value

	def GetQ1Value(self,state,action):
		value=0.0
		if state in self.q1.keys():
			value=self.q1[(state,action)]
		return value

	def GetActionValue(self,state,action):
		q0=0.0
		q1=0.0
		if state in self.q0.keys():
			q0=self.q0[(state,action)]
		if state in self.q1.keys():
			q1=self.q1[state,action]
		return q0+q1

	def ActionFromPolicy(self,state,epsilon=0.1):
		if random.random()>epsilon:
			action=np.argmax([self.GetActionValue(state,a) for a in range(self.action_n)])
		else:
			action=random.randint(0,self.action_n-1)
		return action

	def Learn(self,state,action,reward,next_state,done):
		if np.random.randint(2):
			self.q0,self.q1=self.q1,self.q0
		A=np.argmax([self.GetQ0Value(next_state,i) for i in range(self.action_n)])
		U=reward+self.gamma*self.GetQ1Value(next_state,A)*(1-done)
		td_error=U-self.GetQ0Value(state,action)
		if (state,action) in self.q0.keys():
			self.q0[(state,action)]+=self.learning_rate*td_error
		else:
			self.q0[(state,action)]=self.learning_rate*td_error

	def Train(self,state):
		state=tuple(state.reshape(self.env.grid_size))
		A=self.ActionFromPolicy(state)
		S,R,finish,remark=self.env.step(A)
		S=tuple(S.reshape(self.env.grid_size))
		self.Learn(state,A,R,S,finish)
		self.env.render()
		if R!=0.0:
			print('got %f score'%(R,))
		return R,finish

def main():
	if len(sys.argv)<3:
		return
	envSize=int(sys.argv[1])
	trainCount=int(sys.argv[2])
	roundLimit=envSize*envSize*envSize*envSize
	env=SnakeEnv((envSize,envSize))
	agent=DoubleQLearningAgent(env)
	if len(sys.argv)>=4 and os.path.exists(sys.argv[3]):
		agent.Open(sys.argv[3])
	step=0
	score=0
	print('first:')
	env.render()
	for i in range(trainCount):
		print('train',i)
		R,finish=agent.Train(env.state)
		score+=R
		step+=1
		if finish or step>=roundLimit:
			env.reset()
			print('score:',score)
			print('first:')
			env.render()
			step=0
			score=0
	print('Training complete!')
	if len(sys.argv)>=4:
		agent.Save(sys.argv[3])

if __name__=='__main__':
	main()