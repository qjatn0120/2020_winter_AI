import gym

env = gym.make("GuessingGame-v0")

state = env.reset()
MIN = -1000
MAX = 1000

while 1:
	action = (MIN + MAX) // 2
	state, reward, done, _ = env.step(action)

	print("range {} ~ {}".format(MIN, MAX))
	
	if state == 1:
		MIN = action + 1
	elif state == 3:
		MAX = action - 1

	print("action : {}, state : {}".format(action, state))

	if done:
		break