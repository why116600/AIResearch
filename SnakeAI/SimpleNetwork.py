import numpy as np

def relu(x):
	if x>=0.0:
		return x
	return 0.0

def drelu(x):
	if x>=0.0:
		return 1.0
	return 0.0

RELU=np.vectorize(relu)
dRELU=np.vectorize(drelu)

class Network:
	def __init__(self,shape):
		assert len(shape)>=2
		self.shape=shape
		self.weights=[np.random.normal(0,1,(shape[i],shape[i+1])) for i in range(len(shape)-1)]
	
	def Calculate(self,x):
		assert len(x)==self.shape[0]
		v=np.dot(x,self.weights[0])
		for i in range(1,len(self.weights)):
			x=RELU(v)
			v=np.dot(x,self.weights[i])
		return v

	def Train(self,x,t,rate=0.0001):
		assert len(x)==self.shape[0] and len(t)==self.shape[-1]
		if len(x.shape)==1:
			x=x.reshape(1,x.shape[0])
		if len(t.shape)==1:
			t=t.reshape(1,t.shape[0])
		X=[x,]
		y=np.dot(x,self.weights[0])
		for i in range(1,len(self.weights)):
			x=RELU(y)
			X.append(x)
			y=np.dot(x,self.weights[i])
		delta=t-y
		#print 'x:',X[0]
		#print 'delta:',delta
		for i in range(len(self.weights)-1,-1,-1):
			dw=np.dot(X[i].transpose(),delta)
			#print 'now weight:',self.weights[i]
			#print 'dw[%d]:'%(i,),dw
			if i>0:
				dr=dRELU(np.dot(X[i-1],self.weights[i-1]))
				delta=np.dot(delta,self.weights[i].transpose())*dr
			self.weights[i]+=dw*rate

if __name__=='__main__':
	SIZE=8
	targetw=np.random.normal(0,10,SIZE)
	nn=Network([SIZE,1])
	for i in range(100):
		x=np.random.normal(0,10,SIZE)
		y=np.dot(x,targetw)
		nn.Train(x,np.array([y,]))
	print('target:',targetw)
	print('trained:',nn.weights[0])
