import gym
import random
import torch
import torch.nn as nn
from copy import deepcopy

env = gym.make("Acrobot-v1")
env._max_episode_steps = 1000

alpha = 0.0001 # learning rate
gamma = 0.95 # discount factor
max_episode = 1000000 # the number of epsiodess
epsilon = 0.05 # epsilon for e-greedy
seed = 6321 # seed

env.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

class DQN(nn.Module):

	def __init__(self):
		super().__init__()
		self.layer = nn.Sequential(
				nn.Linear(6, 256),
				nn.ReLU(),
				nn.Linear(256, 256),
				nn.ReLU(),
				nn.Linear(256, 3)
			)

		self.history = []

	def forward(self, x):

		out = self.layer(x)
		return out

base_model = DQN()
model = DQN()
model.load_state_dict(base_model.state_dict())
optimizer = torch.optim.Adam(model.parameters(), lr = alpha)

def select_action(state):
	
	action_value = model(state)
	action = torch.argmax(action_value).item() - 1

	return action

def finish_episode():

	loss = []

	for i in range(1000):

		(state, action, reward, done, next_state) = random.choice(model.history)

		state_value = model(torch.FloatTensor([state]))

		state_value = state_value[0][action + 1]

		if done:
			total_reward = reward
		else:
			total_reward = reward + gamma * base_model(torch.FloatTensor([next_state])).max().item()


		loss.append((state_value - torch.tensor([total_reward])) ** 2)

	loss = torch.stack(loss).sum() / len(loss)
	
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()

	return loss.item()

episode_reward = []

for i_episode in range(1, max_episode + 1):

	state = env.reset()

	total_reward = 0

	while(1):

		action = -1

		if random.random() > epsilon:
			action = select_action(torch.FloatTensor(state))
		else:
			action = random.choice([-1, 0, 1])

		next_state, reward, done, _ = env.step(action)

		reward = float(-state[0] - state[0] * state[2] + state[1] * state[3]) - 1

		model.history.append((state, action, reward, done, next_state))

		total_reward += reward

		if len(model.history) == 100000:
			del model.history[0]

		if i_episode % 100 == 0:
			env.render()

		state = next_state

		if done:
			break

	episode_reward.append(total_reward)

	LOSS = finish_episode()
	if len(episode_reward) == 101:
		del episode_reward[0]

	if i_episode % 20 == 0:
		base_model.load_state_dict(model.state_dict())

	print("[Episode {}] Loss : {:.3f}, Reward : {:.3f}, Average reward : {:.3f}".format(i_episode, LOSS, episode_reward[-1], sum(episode_reward) / len(episode_reward)))

	if sum(episode_reward) / len(episode_reward) > -200:
		torch.save(model.state_dict(), "acrobot.pt")
		print("Clear at episode {}".format(i_episode))
		break
