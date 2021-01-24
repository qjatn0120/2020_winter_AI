from time import sleep

def init_list(x, y):
	ret = []
	for j in range(y):
		tmp = []
		for i in range(x): tmp.append([0, 0, 0, 0])
		ret.append(tmp)
	return ret

class Cliff:

	def __init__(self, x, y):
		assert type(x) is int and type(y) is int, "미로의 규격은 정수이어야 합니다."
		assert x > 2, "가로의 길이는 2보다 커야 합니다."
		assert y > 1, "세로의 길이는 2보다 커야 합니다."
		self.x, self.y = x, y
		self.pos = [0, 0]

	def reset(self):
		self.pos = [0, 0]
		return self.pos

	def step(self, Dir):
		# 0 위쪽, 1 오른쪽, 2 아래쪽, 3 왼쪽
		assert type(Dir) is int, "action은 정수여야 합니다"
		assert Dir >= 0 and Dir < 4, "action은 0이상 4미만이어야 합니다."
		dir_x = [0, 1, 0, -1]
		dir_y = [1, 0, -1, 0]

		px, py = self.pos[0] + dir_x[Dir], self.pos[1] + dir_y[Dir]
		if px < 0: px = 0
		if px >= self.x: px = self.x - 1
		if py < 0: py = 0
		if py >= self.y: py = self.y - 1
		reward = -1
		if py == 0 and px > 0 and px < self.x - 1:
			px, py = 0, 0
			reward = -101
		self.pos = [px, py]
		done = px == self.x - 1 and py == 0
		return self.pos, reward, done

	def render(self):
		for y in range(self.y * 2, -1, -1):
			for x in range(self.x * 2 + 1):
				if y % 2 == 0:
					print('-', end = '')
				else:
					if x % 2 == 0:
						print('|', end = '')
					else:
						if self.pos[0] == x // 2 and self.pos[1] == y // 2:
							print('X', end = '')
						elif x // 2 == 0 and y // 2 == 0:
							print('S', end = '')
						elif x // 2 == self.x - 1 and y // 2 == 0:
							print('E', end = '')
						elif y // 2 == 0:
							print('@', end = '')
						else:
							print(' ', end = '')
			print('')

		sleep(1)