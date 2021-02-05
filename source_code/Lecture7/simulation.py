import gym
import torch
import torch.nn as nn

env = gym.make("Acrobot-v1")

class DQN(nn.Module):

	def __init__(self):
		super().__init__()
		self.layer = nn.Sequential(
				nn.Linear(6, 512),
				nn.ReLU(),
				nn.Linear(512, 512),
				nn.ReLU(),
				nn.Linear(512, 512),
				nn.ReLU(),
				nn.Linear(512, 3)
			)

		self.history = []

	def forward(self, x):

		out = self.layer(x)
		return out

model = DQN()
model.load_state_dict(torch.load("acrobot_1e-4.pt"))

def select_action(state):
	
	action_value = model(state)
	action = torch.argmax(action_value).item() - 1

	return action

for i in range(10):

	state = env.reset()

	while 1:

		action = select_action(torch.FloatTensor(state))

		state, reward, done, _ = env.step(action)

		env.render()

		if done:
			break