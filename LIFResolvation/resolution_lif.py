import math
import numpy as np

EPSCInitialValue_=1.359141;
IPSCInitialValue_=1.359141;
RefractoryCounts_=20;
P11_ex_=0.851229;
P21_ex_=0.095123;
P22_ex_=0.951229;
P31_ex_=0.000019;
P32_ex_=0.000388;
P11_in_=0.851229;
P21_in_=0.095123;
P22_in_=0.951229;
P31_in_=0.000019;
P32_in_=0.000388;
P30_=0.000398;
P33_=0.990050;
expm1_tau_m_=-0.009950;
weighted_spikes_ex_=0.000000;
weighted_spikes_in_=0.000000;

def IterationLIF(inputI,nInter,mI_e=376.0,V_init=-70.0,V_reset=-70.0,V_th=-55.0,timestep=0.1):
	ref=0.0
	mp=V_init-V_reset
	mI_ex_=0.0;
	mdI_ex_=0.0;
	mI_in_=0.0;
	mdI_in_=0.0;
	result=[V_init,]
	if inputI>0.0:
		mdI_ex_ +=EPSCInitialValue_ * inputI;
	else:
		mdI_in_ +=IPSCInitialValue_ * inputI;
	for i in range(nInter):
		if ref<=0.0:
			mp = P30_ * mI_e  + P31_ex_ * mdI_ex_ + P32_ex_ * mI_ex_ +P31_in_ * mdI_in_ + P32_in_ * mI_in_ + expm1_tau_m_ * mp + mp
		else:
			ref-=timestep
		mI_ex_ = P21_ex_ * mdI_ex_ + P22_ex_ * mI_ex_
		mdI_ex_ *= P11_ex_
		mI_in_ = P21_in_ * mdI_in_ + P22_in_ * mI_in_
		mdI_in_ *= P11_in_

		
		V=mp+V_reset
		if V>V_th:
			mp=mV_reset
			ref=2.0
		result.append(V)
	return result

def ResolutionLIF(inputI,length,mI_e=376.0,V_init=-70.0,V_reset=-70.0,V_th=-55.0,timestep=0.1):
	a_ex=P22_ex_
	b_ex=0.0
	d=0.0
	if inputI>0.0:
		b_ex=P21_ex_*EPSCInitialValue_*inputI
		d=P31_ex_*inputI
	q_ex=P11_ex_

	a_in=P22_in_
	b_in=0.0
	f=0.0
	if inputI<0.0:
		b_ex=P21_in_*IPSCInitialValue_*inputI
		f=P31_in_*inputI
	q_in=P11_in_

	c=P30_ * mI_e
	e=P32_ex_
	g=P32_in_
	h=expm1_tau_m_+1.0

	print('h=%f,a_in=%f,q_in=%f'%(h,a_in,q_in))
	init=np.zeros(7)
	init[0]=c/(h-1.0)
	weight=np.array([c/(h-1.0),d*q_ex/(h-q_ex),b_ex*e*a_ex/(h-a_ex),-q_ex*q_ex*b_ex*e/(h-q_ex)/(a_ex-q_ex),f*q_in/(h-q_in),b_in*g*a_in/(h-a_in),-q_in*q_in*b_in*g/(h-q_in)/(a_in-q_in)])
	qs=np.array([1.0,q_ex,a_ex,q_ex,q_in,a_in,q_in])
	#return [math.pow(h,t)*c/(h-1.0)-c/(h-1.0)+V_reset for t in range(length)]
	return [math.pow(h,t)*np.sum(init)-np.dot(weight,np.power(qs,t))+V_reset for t in range(length)]
		

if __name__=='__main__':
	r1=IterationLIF(1.0,10)
	print(r1)
	r2=ResolutionLIF(1.0,10)
	print(r2)
