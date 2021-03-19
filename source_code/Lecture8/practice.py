"""

cartpole 문제를 REINFORCE 알고리즘을 이용하여 학습시키는 연습문제 입니다.

카트를 왼쪽, 오른쪽으로 움직여서 최대한 오래 막대가 기울어지지 않도록 하는 것이 목표입니다.

이 문제에서는 state를 총 4개의 실수로 나타냅니다.

4개의 실수는 각각 카트의 위치, 카트의 속력, 막대의 각도, 막대의 각속도를 나타냅니다.

이 문제에서는 0과 1의 2개의 action이 가능합니다.

action 0은 카트를 왼쪽으로 미는 것, action 1은 카트를 오른쪽으로 미는 것을 의미합니다.

"""

import gym
import torch
import torch.nn as nn
# model에서 얻은 확률에 따라 action을 선택하고, log probability를 계산할 때 사용합니다.
from torch.distributions import Categorical
import random

env = gym.make("CartPole-v1") # cartpole 환경을 가져옵니다.
alpha = 1e-5 # learning rate
gamma = 0.95 # discount factor
max_episode = 10000000 # max episodes
SAVE = True # save model to file if True
LOAD = True # load model from file if True
env._max_episode_steps = 1000000 # max step per episode의 제한을 제거합니다.

# REINFORCE를 위한 neural network를 설계합니다.
# state를 input으로 하고, action을 수행할 확률을 output으로 합니다.

class REINFORCE(nn.Module):

	def __init__(self):

		# 4 -> 256 -> 256 -> 256 -> 2의 layer를 가진 neural network를 설계합니다.
		# 마지막 output에 sofmax 함수를 처리해서 각 action을 수행할 확률의 합이 1이 되도록 합니다.
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

		self.action_history = [] # episode에서 수행했던 action의 log probability를 기록합니다.
		self.reward_history = [] # episode에서 얻은 reward를 기록합니다.

	def forward(self, x):

		action_prob = self.layer(x) # layer로부터 action probability를 가져옵니다.
		return action_prob

model = REINFORCE() # model을 가져옵니다.

if LOAD: # 파일로부터 model을 불러옵니다.
	model.load_state_dict(torch.load("model.pt"))

# Adam optimizer를 정의합니다.
optimizer = torch.optim.Adam(model.parameters(), lr = alpha)

# state를 입력으로 받아 action을 선택하는 함수입니다.
def select_action(state):

	# 모델로부터 각 action을 수행할 확률을 계산합니다.
	action_prob = model(torch.FloatTensor(state))

	# torch의 Categorical을 가져옵니다.
	m = Categorical(action_prob)

	# Categorical을 이용해 action probability에 따라 action을 선택합니다.
	action = m.sample()

	# action의 log probability를 기록합니다.
	model.action_history.append(m.log_prob(action))

	# action이 tensor 타입이므로, int 타입으로 변환해 반환합니다.
	return action.item()

# episode가 끝났을 때, 기록을 바탕으로 학습하는 함수입니다.
def finish_episode():

	# reward 기록을 바탕으로 total reward를 계산합니다.
	R = 0
	returns = []
	for r in model.reward_history[::-1]:
		R = r + R * gamma
		returns.append(R)
	returns.reverse()
	returns = torch.tensor(returns)

	"""
	total reward를 normalization하면
	상대적으로 안좋은 행동은 음수의 보상을 얻고
	상대적으로 좋은 생동은 양수의 보상을 얻으므로
	학습이 더욱 빨라집니다.
	"""
	# returns = (returns - returns.mean()) / returns.std()

	# reward function을 계산합니다.
	loss = []

	for log_prob, reward in zip(model.action_history, returns):

		loss.append(-log_prob * reward)

	loss = torch.stack(loss).sum()

	# reward function으로 학습을 진행합니다.
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()

	# 기록을 전부 지웁니다.
	del model.action_history[:]
	del model.reward_history[:]

# 최근 100 episode의 total reward를 저장할 리스트를 정의합니다.
total_reward = []

# episode를 진행합니다.
for i_episode in range(1, max_episode + 1):

	# environment를 초기화합니다.
	state = env.reset()

	# 해당 episode에서 얻은 총 reward를 기록할 변수를 정의합니다.
	episode_reward = 0

	# episode가 끝날 때까지 진행합니다.
	while 1:

		# action을 선택합니다.
		action = select_action(state)

		# action을 수행합니다.
		state, reward, done, _ = env.step(action)

		# 얻은 reward를 기록합니다.
		model.reward_history.append(reward)

		# 이번 episode에서 얻은 총 reward를 계산합니다.
		episode_reward += reward

		# 만약 episode가 끝났다면 episode를 종료합니다.
		if done:
			break

	# 이번 episode의 기록을 바탕으로 학습을 진행합니다.
	finish_episode()

	# 이번 episode에서 얻은 총 보상을 기록합니다.
	total_reward.append(episode_reward)

	# 최근 100 episode의 총 보상만 기록합니다.
	if len(total_reward) == 101:
		del total_reward[0]

	# 학습 내역을 출력합니다.
	if i_episode % 10 == 0:
		print("[Episode {}] Reward : {}, Average Reward : {}".format(i_episode, episode_reward, sum(total_reward) / len(total_reward)))
		# 모델을 저장합니다.
		if SAVE:
			torch.save(model.state_dict(), "model.pt")