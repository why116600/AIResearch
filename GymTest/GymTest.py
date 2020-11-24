import gym

if __name__=='__main__':
	env=gym.make('MountainCar-v0')
	print('observation space:',env.observation_space)
	print('action space:',env.action_space)
	print('observation range:{} - {}'.format(env.observation_space.low,env.observation_space.high))
	print('action count:',env.action_space.n)
