from random import random
from random import choice
from copy import deepcopy
from time import sleep

class Maze:

	def __init__(self, x, y, rate):
		self.size_x = x
		self.size_y = y
		self.wall_rate = rate
		self.maze = self.new_maze(x, y)
		while not self.solvable():
			for i in range(self.size_y):
				del self.maze[i][:]
			del self.maze[:]
			self.maze = self.new_maze(x, y)
		pos_list = []
		for i in range(self.size_y):
			for j in range(self.size_x):
				if not self.maze[i][j]:
					pos_list.append((j, i))
		self.end_x, self.end_y = choice(pos_list)

	def OutOfRange(self, x, y):
		return x < 0  or x >= self.size_x or y < 0 or y >= self.size_y

	def new_maze(self, x, y):
		ret = []
		for i in range(y):
			tmp = []
			for j in range(x):
				if random() < self.wall_rate:
					tmp.append(1)
				else:
					tmp.append(0)
			ret.append(tmp)
		return ret

	def solvable(self):
		x, y = -1, -1
		dir_x = [0, 1, 0, -1]
		dir_y = [1, 0, -1, 0]
		for i in range(self.size_y):
			for j in range(self.size_x):
				if self.maze[i][j]:
					continue
				x, y = j, i
		visit = deepcopy(self.maze)
		queue = [(x, y)]
		visit[y][x] = 1
		while len(queue):
			x, y = queue[0]
			del queue[0]
			for Dir in range(4):
				px = x + dir_x[Dir]
				py = y + dir_y[Dir]
				if self.OutOfRange(px, py) or visit[py][px]:
					continue
				queue.append((px, py))
				visit[py][px] = 1

		for y in range(self.size_y):
			for x in range(self.size_x):
				if not visit[y][x]:
					return False
		return True

	def display_maze(self):
		for y in range(self.size_y * 4 + 1):
			if y % 4 == 0:
				for i in range(self.size_x * 4 + 1): print('-', end = '')
			else:
				for x in range(self.size_x * 4 + 1):
					if x % 4 == 0: print('|', end = '')
					elif x % 4 == 2 and y % 4 == 2 and self.maze[y // 4][x // 4]: print('@', end = '')
					elif x % 4 == 2 and y % 4 == 2 and x // 4 == self.end_x and y // 4 == self.end_y: print('O', end = '')
					else: print(' ', end = '')
			print('')
		sleep(0.5)

	def display_pi(self, pi):
		for y in range(self.size_y * 4 + 1):
			if y % 4 == 0:
				for i in range(self.size_x * 4 + 1): print('-', end = '')
			else:
				for x in range(self.size_x * 4 + 1):
					if x % 4 == 0: print('|', end = '')
					elif x % 4 == 2 and y % 4 == 1 and pi[y // 4][x // 4][0]: print('^', end = '')
					elif x % 4 == 2 and y % 4 == 3 and pi[y // 4][x // 4][1]: print('v', end = '')
					elif x % 4 == 1 and y % 4 == 2 and pi[y // 4][x // 4][2]: print('<', end = '')
					elif x % 4 == 3 and y % 4 == 2 and pi[y // 4][x // 4][3]: print('>', end = '')
					elif x % 4 == 2 and y % 4 == 2 and self.maze[y // 4][x // 4]: print('@', end = '')
					elif x % 4 == 2 and y % 4 == 2 and x // 4 == self.end_x and y // 4 == self.end_y: print('O', end = '')
					else: print(' ', end = '')
			print('')
		sleep(0.5)
	def possible_state(self):
		ret = []
		for i in range(self.size_y):
			for j in range(self.size_x):
				if self.maze[i][j]: continue
				ret.append((j, i))
		return ret

	def step(self, x, y, action): # action 0 = N 1 = S 2 = W 3 = E
		if x == self.end_x and y == self.end_y: return (x, y), 0, True
		if self.OutOfRange(x, y): return "out of range"
		if action == 0 and y != 0 and not self.maze[y - 1][x]: y -= 1
		if action == 1 and y != self.size_y - 1 and not self.maze[y + 1][x]: y += 1
		if action == 2 and x != 0 and not self.maze[y][x - 1]: x -= 1
		if action == 3 and x != self.size_x - 1 and not self.maze[y][x + 1]: x += 1
		done = x == self.end_x and y == self.end_y
		return (x, y), -1, done