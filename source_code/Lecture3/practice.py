"""
연습문제

임의의 미로를 DP로 해결해봅시다.
초기 policy는 random policy라고 가정합니다.
discount factor gamma는 1이라고 가정합니다.
policy iteration을 반복해가면서 optimal policy를 찾습니다.

env = Maze(x, y, rate) 함수로 새 미로를 만듭니다. 이 때, rate는 벽의 등장 비율을 의미합니다.
너비가 x, 높이가 y인 미로를 생성합니다. 도착점은 랜덤으로 생성됩니다.
이 떄, 벽이 아닌 모든 지점에서 도착점으로 이동할 수 있음이 보장됩니다.

env.possible_state() 함수로 벽이 존재하지 않는 좌표를 리스트로 받을 수 있습니다.
[(x0, y0), (x1, y1), ...]의 형태로 반환됩니다.

env.step(x, y, action)은 (x, y) state에서 action행동을 취하는 함수입니다.
state, reward, done을 반환합니다.
state는 step 이후의 state로 (x, y)에 해당합니다.
만약 도착점에서 step을 시도한 경우, reward = 0입니다. 그 이외의 경우는 reward = -1입니다.
만약 done이 true라면 terminal state라는 뜻입니다.
만약 x, y가 미로의 범위를 벗어나거나, 벽이라면 "out of range"를 반환합니다.

env.display_maze() 함수를 호출하면 미로가 출력됩니다.
env.display_pi(pi) 함수를 호출하면, 양의 확률로 행해지는 action을 출력합니다.
"""

from maze import Maze
from copy import deepcopy

# 상수
size_x = 7 # 미로의 너비
size_y = 7 # 미로의 너비
wall_rate = 0.3 # 벽의 등장 비율
K = 500 # policy evaluation에서의 반복 횟수
max_Iter = 10 # policy iteration의 반복 횟수
eps = 1e-5 # 매우 작은 수, |x - y| < eps라면 x == y로 취급한다.

# 기본적인 환경을 세팅합니다.

env = Maze(size_x, size_y, wall_rate)
state = env.possible_state()

# policy를 저장할 list
pi = []

# pi를 초기화합니다.
for y in range(size_y):
	tmp = []
	for x in range(size_x):
		tmp.append([0, 0, 0, 0])
	pi.append(tmp)

# possible state의 policy를 random policy로 setting합니다.
for x, y in state:
	pi[y][x] = [0.25, 0.25, 0.25, 0.25]

# value function을 저장할 list
v = []

# v를 초가화합니다.
for y in range(size_y):
	tmp = []
	for x in range(size_x):
		tmp.append(0)
	v.append(tmp)

for Iter in range(1, max_Iter + 1):
	# policy evaluation

	# k번 반복합니다.
	for k in range(K):

		# k+1 value function을 저장할 공간을 만듭니다.
		next_v = deepcopy(v)

		# 모든 가능한 state에 대해서
		for x, y in state:

			# value function을 초기화합니다.
			next_v[y][x] = 0

			# 모든 action에 대해 step을 진행하여 next_v를 계산합니다.
			for action in range(4):

				# action을 합니다.
				(px, py), reward, done = env.step(x, y, action)
				
				# next_v를 업데이트 합니다.
				next_v[y][x] += pi[y][x][action] * (reward + v[py][px])

		# value function을 갱신합니다.

		v = deepcopy(next_v)

	# policy improvement
	
	# 모든 가능한 state에 대해서
	for x, y in state:

		# max v값을 구합니다.
		max_v = -1e9

		for action in range(4):
			(px, py), _, _ = env.step(x, y, action)
			max_v = max([max_v, v[py][px]])

		# max v값을 가지게 하는 action의 개수를 구합니다.
		count = 0

		for action in range(4):
			(px, py), _, _ = env.step(x, y, action)
			if abs(v[py][px] - max_v) < eps:
				count += 1

		# policy를 update합니다.
		for action in range(4):
			(px, py), _, _ = env.step(x, y, action)
			if abs(v[py][px] - max_v) < eps:
				pi[y][x][action] = 1 / count
			else:
				pi[y][x][action] = 0

# DP 계산 결과를 출력합니다.
env.display_maze()
env.display_pi(pi)
