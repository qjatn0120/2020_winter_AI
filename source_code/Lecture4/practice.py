"""
이번 연습문제에서는 Sarsa와 Q-learning을 구현하여 cliff 문제를 해결합니다.

cliff.py의 Cliff class에는 cliff-walking 문제가 구현되어 있습니다.
env = Cliff(x, y)로 가로가 x, 세로가 y인 cliff-walking 환경을 만듭니다.
env.reset() 함수를 이용하여 환경을 초기화합니다.
state = env.reset()의 형태로 초기 환경을 얻을 수 있습니다.
env.step(a) 함수를 이용하여 action을 취할 수 있습니다.
a = 0 위쪽, a = 1 오른쪽, a = 2 아래쪽, a = 3 왼쪽 을 의미합니다.
env.step(a) 함수는 action을 취한 후의 state, reward 그리고 종료 여부를 반환합니다.
next_state, reward, done = env.step(a)의 형태로 사용할 수 있습니다.
env.render() 함수를 사용하여 진행사항을 출력해볼 수 있습니다.
env.render() 함수는 sublime text가 과부하가 걸리는 것을 막기 위해
1초의 지연시간을 두고 출력합니다. 따라서 무분별하게 env.render()를 사용하면
학습 속도가 상당히 느려집니다. 학습 종료 후에 시험용으로 사용하시길 바랍니다.

cliff.py의 init_list 함수를 이용해 초기상태의 Q를 가져올 수 있습니다.
모든 값은 0으로 초기화됩니다.
Q = init_list(y, x) 꼴로 사용할 수 있습니다.
"""

# cliff.py에서 cliff walking 게임 환경을 가져옵니다.
from random import random, choice
from cliff import Cliff, init_list

# 상수
size_x = 10 # cliff의 가로 길이
size_y = 5 # cliff의 세로 길이
gamma = 0.9 # discount rate
alpha = 0.001 # learning rate
i_episode = 100000 # 총 episode의 수
epsilon = 0.05 # e-greedy 상수

env = Cliff(size_x, size_y)

# value function을 초기화합니다.
# Q[y][x][a] = total reward 꼴로 저장합니다.
Q1 = init_list(size_x, size_y)
Q2 = init_list(size_x, size_y)

# state pos에서 action a를 greedy policy로 선택합니다.

def greedy(q, pos):
	max_Q = -100000000 # Q값의 max값을 저장할 변수
	max_action = -1 # Q값이 max가 되도록 하는 action

	# argmax Q(state, action)을 계산한다.
	for action in range(4):
		
		# max_Q 갱신
		if max_Q < q[pos[1]][pos[0]][action]:
			max_Q = q[pos[1]][pos[0]][action]
			max_action = action

	return max_action

# state pos에서 action a를 e-greedy policy로 선택합니다.

def e_greedy(q, pos):
	max_action = greedy(q, pos) # Q값이 max가 되도록 하는 action

	if random() < epsilon: # e의 확률로 랜덤 action을 취한다.
		return choice([0, 1, 2, 3])
	else: # 1-e의 확률로 greedy action을 취한다.
		return max_action



# Sarsa 알고리즘을 사용하여 value function을 update합니다.

def Sarsa(q):
	for episode in range(1, i_episode + 1):

		# initialize state
		state = env.reset()

		# choose a from s using e-policy
		action = e_greedy(q, state)

		while 1:

			# take action A, observe R, S'
			next_state, reward, done = env.step(action)

			# choose A' from S' using e-policy
			next_action = e_greedy(q, next_state)

			# update value function
			q[state[1]][state[0]][action] += alpha * (reward + gamma * q[next_state[1]][next_state[0]][next_action] - q[state[1]][state[0]][action])

			# S <- S', A <- A'
			state = next_state
			action = next_action

			# until S is terminal
			if done:
				break

# Q-learning 알고리즘을 사용하여 value function을 update합니다.
def Q_learning(q):
	for episode in range(1, i_episode + 1):

		# initialize state
		state = env.reset()

		while 1:

			# choose A from S using e-policy
			action = e_greedy(q, state)

			# Take action A, observe R, S'
			next_state, reward, done = env.step(action)

			# update value function
			q[state[1]][state[0]][action] += alpha * (reward + gamma * q[next_state[1]][next_state[0]][greedy(q, next_state)] - q[state[1]][state[0]][action])

			# S <- S'
			state = next_state

			# until S is terminal
			if done:
				break

# render하면서 게임을 play합니다. Q function 학습이 끝나고 테스트할 때 사용합니다.

def play(q):
	state = env.reset()
	env.render()
	while 1:
		action = greedy(q, state)
		state, reward, done = env.step(action)
		env.render()
		if done:
			break

Sarsa(Q1)
Q_learning(Q2)
play(Q1)
play(Q2)