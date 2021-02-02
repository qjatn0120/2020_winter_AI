import gym
import random

env = gym.make("FrozenLake-v0", is_slippery = False)

action_size = env.action_space.n
state_size = env.observation_space.n

Qtable = []
for i in range(state_size):
	Qtable.append([0] * action_size)

alpha = 0.01 # learning rate
gamma = 1 # discount factor
epsilon = 1
max_episode = 10000000 # 무한히 실행한다.

history = []

for i_episode in range(1, max_episode + 1):
	state = env.reset()

	while 1:

		action = -1

		if random.random() > epsilon:
			max_action_value = -100000
			for i in range(4):
				if Qtable[state][i] > max_action_value:
					max_action_value = Qtable[state][i]
					action = i
		else:
			action = random.choice([0, 1, 2, 3])

		next_state, reward, done, _ = env.step(action)

		Qtable[state][action] += alpha * (reward + gamma * max(Qtable[next_state]) - Qtable[state][action])
		state = next_state

		if reward == 0:
			if done:
				reward = -1000
			else:
				reward = -1

		if done:
			break

	if reward == 1:
		history.append(1)
	else:
		history.append(0)

	if len(history) == 10001:
		del history[0]

	if i_episode % 1000 == 0:
		print("완주율 : {:.2f}%, epsilon : {:.3f}".format(sum(history) / len(history) * 100, epsilon))
		epsilon *= 0.95