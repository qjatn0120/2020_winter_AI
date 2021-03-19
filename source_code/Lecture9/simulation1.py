import gym
import torch
import torch.nn as nn
from torch.distributions import Categorical

actor_hidden_layer = [256, 256]
critic_hidden_layer = [256, 256, 256]
state_number = 4
action_number = 2

class Actor_Critic_model(nn.Module):

	def __init__(self):
		super().__init__()

		self.actor_layer = nn.Sequential(
				nn.Linear(state_number, actor_hidden_layer[0]),
				nn.ReLU(),
				nn.Linear(actor_hidden_layer[0], actor_hidden_layer[1]),
				nn.ReLU(),
				nn.Linear(actor_hidden_layer[1], action_number),
				nn.Softmax(dim = 0)
			)

		self.critic_layer = nn.Sequential(
				nn.Linear(state_number, critic_hidden_layer[0]),
				nn.ReLU(),
				nn.Linear(critic_hidden_layer[0], critic_hidden_layer[1]),
				nn.ReLU(),
				nn.Linear(critic_hidden_layer[1], critic_hidden_layer[2]),
				nn.ReLU(),
				nn.Linear(critic_hidden_layer[2], 1)
			)


	def forward(self, x):
		action_prob = self.actor_layer(x)
		return action_prob

def select_action(state):
	action_prob = model(torch.FloatTensor(state))
	m = Categorical(action_prob)
	action = m.sample()
	return action.item()

env = gym.make("CartPole-v1")
model = Actor_Critic_model()
model.load_state_dict(torch.load("model.pt"))

for i in range(10):

	state = env.reset()
	while 1:
		action = select_action(state)
		state, reward, done, _ = env.step(action)
		env.render()
		if done:
			break
