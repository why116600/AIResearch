import numpy as np
import gym
from gym.spaces import Discrete

class SnakeEnv(gym.Env):
	SIZE=100

	def __init__(self,ladder_num,dices):
		self.ladder_num=ladder_num
		self.dices=dices
		self.ladders=dict(np.random.randint(1,self.SIZE,size=(self.ladder_num,2)))
		self.observation_space=Discrete(self.SIZE+1)
		self.action_space=Discrete(len(dices))

		for k,v in self.ladders.items():
			self.ladders[v]=k
		self.pos=1

	def reset(self):
		self.pos=1
		return self.pos

	def step(self,a):
		step=np.random.randint(1,self.dices[a]+1)
		self.pos+=step
		if self.pos==100:
			return 100,100,1,{}
		elif self.pos>100:
			self.pos=200-self.pos
		if self.pos in self.ladders:
			self.pos=self.ladders[self.pos]
		return self.pos,-1,0,{}

	def reward(self,s):
		if s==100:
			return 100
		else:
			return -1

	def render(self):
		pass

def list_policy(env,policy):
	state=env.reset()
	ret=0
	while True:
		act=policy[state]
		state,reward,terminate, _ =env.step(act)
		ret+=reward
		if terminate:
			break
	return ret

class TableAgent(object):
	def __init__(self,env):
		self.s_len=env.observation_space.n
		self.a_len=env.action_space.n
		
		self.r=[env.reward(s) for s in range(self.s_len)]
		self.pi=np.array([0 for s in range(self.s_len)])
		self.p=np.zeros([self.a_len,self.s_len,self.s_len],dtype=np.float)

		ladder_move=np.vectorize(lambda x: env.ladders[x] if x in env.ladders else x)

		for i, dice in enumerate(env.dices):
			prob=1.0/dice
			for src in range(1,100):
				step=np.arange(dice)
				step+=src
				step=np.piecewise(step,[step>100,step<=100],[lambda x: 200-x,lambda x: x])
				step=ladder_move(step)
				for dst in step:
					self.p[i,src,dst]+=prob
		self.p[:,100,100]=1
		self.value_pi=np.zeros((self.s_len,))
		self.value_q=np.zeros((self.s_len,self.a_len))
		self.gamma=0.8

	def play(self,state):
		return self.pi[state]

def policy_evaluation(agent,max_iter=-1):
	iteration=0
	while True:
		iteration+=1
		new_value_pi=agent.value_pi.copy()
		for i in range(1,agent.s_len):
			value_sas=[]
			ac=agent.pi[i]
			transition=agent.p[ac,i,:]
			value_sa=np.dot(transition,agent.r+agent.gamma*agent.value_pi)
			new_value_pi[i]=value_sa
		diff=np.sqrt(np.sum(np.power(agent.value_pi-new_value_pi,2)))
		if diff<1e-6:
			break
		else:
			agent.value_pi=new_value_pi
			#print 'value:',new_value_pi
		if iteration==max_iter:
			break

def policy_improvement(agent):
	new_policy=np.zeros_like(agent.pi)
	for i in range(1,agent.s_len):
		for j in range(0,agent.a_len):
			agent.value_q[i,j]=np.dot(agent.p[j,i,:],agent.r+agent.gamma*agent.value_pi)
			max_act=np.argmax(agent.value_q[i,:])
			new_policy[i]=max_act
			print 'policy:',max_act
	if np.all(np.equal(new_policy,agent.pi)):
		return False
	else:
		agent.pi=new_policy
		return True

def policy_iteration(agent):
	iteration=0
	while True:
		iteration+=1
		policy_evaluation(agent)
		ret=policy_improvement(agent)
		if not ret:
			break


if __name__=='__main__':
	env=SnakeEnv(0,[3,6])
	agent=TableAgent(env)
	policy_iteration(agent)
	print agent.pi
