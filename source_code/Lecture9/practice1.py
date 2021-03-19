"""
Actor Critic을 이용하여 CartPole 문제를 해결합니다.
Actor : policy를 계산합니다.
Critic : actoin value를 계산합니다.
"""

import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.nn.init import kaiming_uniform_

# 기본적인 상수들을 정의합니다.
env = gym.make("CartPole-v1")
env._max_episode_steps = 10000
actor_hidden_layer = [256, 256] # actor model의 hidden layer의 노드의 수를 정의합니다.
critic_hidden_layer = [256, 256, 256] # critic model의 hidden layer의 노드의 수를 정의합니다.
state_number = 4 # state의 개수를 정의합니다.
action_number = 2 # action의 개수를 정의합니다.
actor_learning_rate = 1e-3 # actor layer learning rate
critic_learning_rate = 1e-4 # ciritc layer learning rate
gamma = 0.95 # discount factor
max_episode = 100000 # max episodes
SAVE = True # save model to file
LOAD = True # load model from file
RENDER = False # render environment if true
render_interval = 500 # if RENDER is true, render environment every "render_interval" episodes
log_interval = 10 # print log every "log_interval" episodes
init_bias = 0

def init_weights(m):
	if type(m) == nn.Linear:
		kaiming_uniform_(m.weight, nonlinearity='relu')
		m.bias.data.fill_(init_bias)

# Actor Critic NN을 설계합니다.
class Actor_Critic_model(nn.Module):

	def __init__(self):
		super().__init__()

		# actor layer를 정의합니다.
		self.actor_layer = nn.Sequential(
				nn.Linear(state_number, actor_hidden_layer[0]),
				nn.ReLU(),
				nn.Linear(actor_hidden_layer[0], actor_hidden_layer[1]),
				nn.ReLU(),
				nn.Linear(actor_hidden_layer[1], action_number),
				nn.Softmax(dim = 0)
			)

		# critic layer를 정의합니다.
		self.critic_layer = nn.Sequential(
				nn.Linear(state_number, critic_hidden_layer[0]),
				nn.ReLU(),
				nn.Linear(critic_hidden_layer[0], critic_hidden_layer[1]),
				nn.ReLU(),
				nn.Linear(critic_hidden_layer[1], critic_hidden_layer[2]),
				nn.ReLU(),
				nn.Linear(critic_hidden_layer[2], 1)
			)


		# action에 대한 정보 (log_prob, state_value)를 저장할 buffer를 정의합니다.
		self.action_history = []

		# reward에 대한 정보 (reward, done, next_state_value)를 저장할 buffer를 정의합니다.
		self.reward_history = []

		self.actor_layer.apply(init_weights)

	def forward(self, x):

		# policy를 얻습니다.
		action_prob = self.actor_layer(x)

		# state value를 얻습니다.
		state_value = self.critic_layer(x)

		# policy와 action value를 반환합니다.
		return action_prob, state_value

# 먼저 모델을 가져옵니다.
model = Actor_Critic_model()

# 파일에서 모델을 불러옵니다.
if LOAD:
	model.load_state_dict(torch.load("model.pt"))

# Adam optimier를 가져옵니다.
actor_optimizer = optim.Adam(model.actor_layer.parameters(), lr = actor_learning_rate)
critic_optimizer = optim.Adam(model.critic_layer.parameters(), lr = critic_learning_rate)

# state를 입력받아 action을 선택하는 함수입니다.
def select_action(state):

	# model을 이용하여 policy와 state value를 계산합니다.
	action_prob, state_value = model(torch.FloatTensor(state))

	# policy를 이용해 Categorical을 생성합니다.
	m = Categorical(action_prob)

	# Categorical을 이용해 action probability에 따라 action을 선택합니다.
	action = m.sample()

	# action의 log probability를 계산합니다.
	log_prob = m.log_prob(action)

	# action history에 정보를 기록합니다.
	model.action_history.append((log_prob, state_value))

	return action.item()

# 학습이 종료된 후 기록을 바탕으로 학습을 진행하는 함수입니다.
def finish_episode():

	# actor loss와 critic loss를 따로 계산합니다.
	actor_loss = []
	critic_loss = []

	for (log_prob, state_value), (reward, done, next_state_value) in zip(model.action_history, model.reward_history):

		# advantage를 계산합니다.
		advantage = reward + gamma * (1 - done) * next_state_value.item() - state_value.item()

		# actor loss를 계산합니다.
		actor_loss.append(-log_prob * advantage)

		# critic loss를 계산합니다.
		critic_loss.append((reward + gamma * (1 - done) * next_state_value.item() - state_value) ** 2)

	# actor loss와 critic loss를 전부 더합니다.
	actor_loss = torch.stack(actor_loss).sum()
	critic_loss = torch.stack(critic_loss).sum()
	loss = actor_loss + critic_loss

	# 학습을 진행합니다.
	actor_optimizer.zero_grad()
	critic_optimizer.zero_grad()
	loss.backward()
	actor_optimizer.step()
	critic_optimizer.step()

	# history buffer를 비웁니다.
	del model.action_history[:]
	del model.reward_history[:]

	# loss를 반환합니다.
	return actor_loss.item(), critic_loss.item()

# main 함수를 정의합니다.
# c, c++의 main함수와 동일한 역할을 한다고 생각하면 됩니다.
def main():

	# episode에서 얻은 total reward를 기록합니다.
	total_reward = []

	# max episode번 에피소드를 진행합니다.
	for i_episode in range(1, max_episode + 1):

		# 먼저 환경을 초기화합니다.
		state = env.reset()

		# 이번 episode에서 얻을 reward를 계산합니다.
		episode_reward = 0

		# 게임이 종료될 때까지 게임을 진행합니다.
		while 1:

			# action을 선택합니다.
			action = select_action(state)

			# action을 취합니다.
			state, reward, done, _ = env.step(action)

			# episode reward를 계산합니다.
			episode_reward += reward

			# next state value를 계산합니다.
			_, state_value = model(torch.FloatTensor(state))

			# reward history에 진행 정보를 저장합니다.
			model.reward_history.append((reward, done, state_value))

			# 환경을 render합니다.
			if RENDER and i_episode % render_interval == 0:
				env.render()

			# 만약 terminal state라면 에피소드를 종료합니다.
			if done:
				break

		# history를 바탕으로 학습을 진행합니다.
		actor_loss, critic_loss = finish_episode()

		# episode reward를 기록합니다.
		total_reward.append(episode_reward)

		# total_reward의 크기를 100으로 유지합니다.
		if len(total_reward) == 101:
			del total_reward[0]

		# log를 출력합니다.
		if i_episode % log_interval == 0:
			print("[Episode {}] actor loss : {:.4f}, critic loss : {:.4f}, Average reward : {:.2f}".format(i_episode, actor_loss, critic_loss, sum(total_reward) / len(total_reward)))
			# 모델을 저장합니다.
			if SAVE:
				torch.save(model.state_dict(), "model1.pt")

		if sum(total_reward) / len(total_reward) >= 1000:
			break

# main 함수를 실행합니다.
# 이렇게 하면 import등으로 인해 외부에서 호출될 경우에는 main이 실행되지 않습니다.
# 자세한 내용은 __name__에 대해 구글링해보시기 바랍니다.
if __name__ == "__main__":
	main()