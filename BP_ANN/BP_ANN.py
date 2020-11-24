import numpy as np

def Relu(x):
	return np.maximum(x,0)

def ReluDerivative(x):
	y=x.copy()
	y[y>0.0]=1.0
	y[y<=0.0]=0.0
	return y

def Sigmoid(x):
	return 1/(1+np.exp(-x))

def SigmoidDerivative(x):
	return Sigmoid(x)*(1-Sigmoid(x))

FunctionMap={'sigmoid':Sigmoid,'relu':Relu}
DerivativeMap={'sigmoid':SigmoidDerivative,'relu':ReluDerivative}


def Calculate(nn,x):
	for item in nn:
		if type(item)==str:
			func=FunctionMap[item]
			x=func(x)
		else:
			x=np.dot(item,x)
	return x

def Train(nn,x,yd,rate):
	steps=[x,]
	for item in nn:
		if type(item)==str:
			func=FunctionMap[item]
			x=func(x)
		else:
			x=np.dot(item,x)
		steps.append(x)
	err=rate*(yd-x)
	for i in range(len(steps)-2,-1,-1):
		if type(nn[i])==str:
			func=DerivativeMap[nn[i]]
			err*=func(steps[i])
		else:
			w=nn[i].transpose()
			nn[i]+=np.dot(err.reshape(len(err),1),steps[i].reshape(1,len(steps[i])))
			err=np.dot(w,err)

if __name__=='__main__':
	Nh=100
	data=[(np.array([0.1,0.1]),0.1),(np.array([0.1,1.0]),1.0),(np.array([1.0,0.1]),1.0),(np.array([1.0,1.0]),0.1)]
	nn=[np.random.randn(Nh,2),'relu',np.random.randn(1,Nh),'sigmoid']
	#trainning
	for i in range(5000):
		Train(nn,data[i%4][0],data[i%4][1],0.1)
	#checking
	for i in range(len(data)):
		print(Calculate(nn,data[i][0]))