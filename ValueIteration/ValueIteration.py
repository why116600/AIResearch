import numpy as np

class MapEnv:
	def __init__(self,node,edge):
		self.state_count=node
		self.action_count=node
		alllength=0.0
		for s,t,r in edge:
			alllength+=r
		self.negative=-alllength
		self.rewards=np.ones((node,node))*(-alllength)
		self.propa=np.zeros((node,node,node))
		self.state_done=np.zeros((node,))
		self.state_done[node-1]=1.0
		self.dijkstra_value=np.ones((node,))*(-alllength)#迪杰斯特拉算法评估状态价值
		self.pos=0
		self.reward=0.0
		for s,t,r in edge:
			if s==t:
				continue
			reward=self.state_done[t]*alllength-r
			self.rewards[s,t]=reward
			self.propa[s,t,t]=1.0
			#if t!=(node-1):
				#self.rewards[s,t]=-r
			reward=self.state_done[s]*alllength-r
			self.rewards[t,s]=reward
			self.propa[t,s,s]=1.0
			#if s!=(node-1):
				#self.rewards[t,s]=-r
				#self.propa[t,s,s]=1.0
		#print('rewards:',self.rewards)

		#迪杰斯特拉算法
		self.dijkstra_value[-1]=0.0
		eternal=set()
		for i in range(node):
			mini=-1
			for j in range(self.dijkstra_value.shape[0]):
				if j not in eternal and (mini<0 or self.dijkstra_value[mini]<self.dijkstra_value[j]):
					mini=j
			if mini<0:
				break
			eternal.add(mini)
			#print('make %d eternal'%(mini,))
			for j in range(node):
				if j in eternal:
					continue
				v=self.dijkstra_value[mini]+self.rewards[j,mini]
				#print('check [%d]:%f-%f'%(j,self.dijkstra_value[j],v))
				if self.propa[j,mini,mini]==1.0 and v>self.dijkstra_value[j]:
					self.dijkstra_value[j]=v
					#print('update [%d]=%f'%(j,self.dijkstra_value[j]))

		print('Dijkstra value:',self.dijkstra_value)

	def step(self,a):
		assert a<self.action_count and a>=0
		self.reward+=self.rewards[self.pos,a]
		if self.propa[self.pos,a,a]==1.0:
			self.pos=a

	def reset(self):
		self.pos=0
		self.reward=0.0


def ValueIteration(env,torlerance,gamma):
	values=np.zeros((env.state_count,))
	k=0
	while True:
		delta=0.0
		new_values=values.copy()
		for s in range(env.state_count):
			allv=env.rewards[s,:]+gamma*np.dot(env.propa[s,:,:],values)
			v=np.max(allv)
			delta=max(delta,abs(v-values[s]))
			new_values[s]=v
		print('iteration',k,':',new_values)
		values=new_values
		k+=1
		if delta<torlerance:
			break
	policy=[0 for i in range(env.state_count)]
	for s in range(env.state_count):
		valueq=env.rewards[s,:]+gamma*np.dot(env.propa[s,:,:],values)
		policy[s]=np.argmax(valueq)
	return policy

if __name__=='__main__':
	map=MapEnv(4,[(0,1,1.0),(1,2,1.0),(0,2,3.0),(1,3,4.0),(2,3,1.0)])
	policy=ValueIteration(map,1e-3,1.0)
	print(policy)

			