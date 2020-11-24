import numpy as np
import random
import sys
import os
from SnakeAI import SnakeEnv
from SnakeAI import Network

class QSnakeAgent:
	def __init__(self,env):
		self.env=env
		self.network=Network(env.grid_size+1)

	def ActionFromPolicy(self,state,IbuSloan=0.2):
		values=[self.network.Calculate(np.append(state.reshape(1,self.env.grid_size),np.array([a,]))) for a in range(4)]
		nAction=4
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

	def Train(self,state,gamma=0.99):
		A=self.ActionFromPolicy(state)
		S,R,finish,remark=self.env.step(A)
		U=R+gamma*np.max([self.network.Calculate(np.append(S.reshape((1,self.env.grid_size)),np.array([a,]))) for a in range(4)])
		print('U=',U)
		xdata=np.append(state.reshape((1,self.env.grid_size)),np.array([A,]))
		ydata=np.array([U,])
		self.network.Train(xdata.reshape((1,self.env.grid_size+1)),ydata.reshape((1,1)))
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
	agent=QSnakeAgent(env)
	if len(sys.argv)>=4 and os.path.exists(sys.argv[3]):
		agent.network.LoadWeight(sys.argv[3])
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
		agent.network.SaveWeight(sys.argv[3])

if __name__=='__main__':
	main()
