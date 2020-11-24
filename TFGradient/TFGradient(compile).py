import tensorflow.compat.v2 as tf
from tensorflow import keras

class CompileError(Exception):
	def __init__(self,msg):
		self.errorinfo=msg

	def __str__(self):
		return self.errorinfo

def rFind(expr,oper):
	circle=0
	for i in range(len(expr)-1,-1,-1):
		if expr[i]==oper and circle==0:
			return i
		if expr[i]=='(':
			circle-=1
		if expr[i]==')':
			circle+=1
	return -1

def Compile(expr,vars):
	if expr in vars.keys():
		return vars[expr]
	if expr[0]=='(' and expr[-1]==')':
		return Compile(expr[1:-1],vars)
	oper=rFind(expr,'-')
	if oper>=0:
		return Compile(expr[0:oper],vars)-Compile(expr[oper+1:],vars)
	oper=rFind(expr,'+')
	if oper>=0:
		return Compile(expr[0:oper],vars)+Compile(expr[oper+1:],vars)
	oper=rFind(expr,'/')
	if oper>=0:
		return Compile(expr[0:oper],vars)/Compile(expr[oper+1:],vars)
	oper=rFind(expr,'*')
	if oper>=0:
		return Compile(expr[0:oper],vars)*Compile(expr[oper+1:],vars)
	oper=rFind(expr,'^')
	if oper>=0:
		return Compile(expr[0:oper],vars)**Compile(expr[oper+1:],vars)
	value=float(expr)
	return tf.Variable(tf.constant(value))

def main():
	x=tf.Variable(tf.constant(0.0))
	vars={'x':x}
	expr=input('input your expression:')
	#y=Compile('(x-10)^2',vars)
	#tape=tf.GradientTape()
	optimizer = tf.optimizers.Adam(1.0)
	for i in range(100):
		with tf.GradientTape() as tape:
			y=Compile(expr,vars)
			#x=tf.Variable(tf.constant(0.0))
			#x=model(tf.convert_to_tensor([[0.0,],],dtype=tf.float32))[0]
			#y=a*x**2+b*x+c
		dx=tape.gradient(y,x)
		print('dx=',dx.numpy())
		print('y=',y.numpy())
		#ddx=[t*tf.constant([-1.0,]) for t in dx]
		optimizer.apply_gradients(zip([dx,], [x,]))
		print('x=',x.numpy())
		

if __name__=='__main__':
	main()
