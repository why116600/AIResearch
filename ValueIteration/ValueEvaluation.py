import numpy as np
from ValueIteration import MapEnv

def NewValue(values,env):
	RetValue=np.zeros(values.shape)
	for i in range(values.shape[0]):
		sum=0.0
		matAV=np.dot(env.propa[i],env.dijkstra_value-env.negative)
		action=np.argmax(matAV)
		if matAV[action]==0.0:
			continue
		nextS=action
		RetValue[i]=env.rewards[i,action]+values[nextS]
	return RetValue

def main():
	print("Start value evaluation")
	map=MapEnv(4,[(0,1,1.0),(0,2,3.0),(1,2,1.0),(2,3,2.0)])
	values=np.zeros((4,))
	for i in range(1000):
		newValue=NewValue(values,map)
		distance=np.dot(newValue-values,newValue-values)
		if distance<4.0:
			break
		print('now values:',newValue)
		print('now distance:',distance)
		values=newValue

if __name__=='__main__':
	main()
