"""
학습시킨 모델을 이용하여 cartpole을 플레이합니다.
"""
import gym
import torch
import torch.nn as nn
from torch.distributions import Categorical

env = gym.make("CartPole-v1")

class REINFORCE(nn.Module):

	def __init__(self):

		super().__init__()
		self.layer = nn.Sequential(
				nn.Linear(4, 256),
				nn.ReLU(),
				nn.Linear(256, 256),
				nn.ReLU(),
				nn.Linear(256, 256),
				nn.ReLU(),
				nn.Linear(256, 2),
				nn.Softmax(dim = 0)
			)

	def forward(self, x):

		action_prob = self.layer(x)
		return action_prob

model = REINFORCE()
model.load_state_dict(torch.load("model.pt"))

def select_action(state):

	action_prob = model(torch.FloatTensor(state))
	m = Categorical(action_prob)
	action = m.sample()
	return action.item()

for i in range(10):
	state = env.reset()
	while 1:
		action = select_action(state)
		state, reward, done, _ = env.step(action)
		env.render()
		if done:
			break