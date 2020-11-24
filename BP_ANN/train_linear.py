import BP_ANN
import numpy as np

if __name__=='__main__':
	N=20
	rightW=np.random.rand(1,N)*2-np.ones((1,N))
	nn=[np.random.randn(1,N),'sigmoid']
	#training
	for i in range(2000):
		x=np.random.rand(N)
		if np.dot(rightW,x)>=0.0:
			y=1.0
		else:
			y=0.1
		BP_ANN.Train(nn,x,y,0.1)
	#checking
	correct=0
	for i in range(100):
		x=np.random.rand(N)
		y=BP_ANN.Calculate(nn,x)
		if np.dot(rightW,x)*(y*2-1)>=0.0:
			correct+=1
	print('correct=%d'%(correct,))
