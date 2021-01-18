"""
덧셈, 곱셈 등의 사칙연산을 지원하는 계산기를 만들어 봅시다.
이번 계산기는 메모리에 최근 계산 식 10개를 저장합니다.
__getitem__ 메서드를 이용하여 계산기의 x번째 계산 식에 접근할 수 있도록 합니다.
__str__ 메서드를 이용하여 계산기 최근 계산 식 10개를 모두 출력할 수 있도록 합니다.
"""

class calculator:

	# calculator 클래스로 인스턴스를 만들 때 자동으로 실행되는 메서드 입니다.
	def __init__(self):

		# 먼저 계산 식을 저장할 메모리(리스트)를 생성합니다.
		self.memory = []

		# 계산기 내부에 값을 저장할 공간 cache를 5개 생성합니다.
		self.cache = [0, 0, 0, 0, 0]

	# self.memory에 계산 식을 저장하는 함수입니다.
	def update_memory(self, string):
		
		# memory의 맨 앞에 계산 식을 추가합니다.
		self.memory.insert(0, string)

		if len(self.memory) == 11: # 만약 메모리의 크기가 10을 초과하면
			del self.memory[-1] # == del self.memory[10], 가장 옛날의 메모리를 삭제합니다.

	# 사칙연산 메서드
	def add(self, x, y):
		result = x + y # 더하기를 계산합니다.
		self.update_memory("{} + {} = {}".format(x, y, result)) # 계산 식을 memory에 추가합니다.
		return result # 계산 결과를 반환합니다.

	def sub(self, x, y):
		result = x - y # 빼기를 계산합니다.
		self.update_memory("{} - {} = {}".format(x, y, result)) # 계산 식을 memory에 추가합니다.
		return result # 계산 결과를 반환합니다.

	def mul(self, x, y):
		result = x * y # 곱하기를 계산합니다.
		self.update_memory("{} * {} = {}".format(x, y, result)) # 계산 식을 memory에 추가합니다.
		return result # 계산 결과를 반환합니다.

	def div(self, x, y):
		result = x / y # 나누기를 계산합니다.
		self.update_memory("{} / {} = {}".format(x, y, result)) # 계산 식을 memory에 추가합니다.
		return result # 계산 결과를 반환합니다.

	# calculator[x]로 x번째 memory에 접근할 수 있도록 __getitem__ 메서드를 구현합니다.
	# calculator[x] == calculator.__getitme__(x)
	def __getitem__(self, x):
		if x < 0 or x >= len(self.memory): # 만약 x가 음수 또는 memory의 크기보다 커서 범위를 벗어났다면
			return "Out of Range" # 계산 식 대신 경고 문구를 반환합니다.
		return self.memory[x] # x번째 계산 식을 반환합니다.

	"""
	print(calc)와 같이 인스턴스를 출력하면 해당 인스턴스의 클래스나 주소 등 우리에게는 필요없는 정보가 출력됩니다.
	따라서 우리는 print(calc)를 하면 메모리의 내용을 출력하도록 하고 싶습니다.
	0. 0번째 메모리
	1. 1번째 메모리
	...
	와 같은 형식으로 출력합니다.
	"""
	def __str__(self):
		ret = "" # 출력할 내용을 ret에 저장합니다.
		for x in range(len(self.memory)): # 0번째 원소부터 마지막 원소까지
			ret += "{}. {}\n".format(x, self.memory[x]) # 형식에 맞게 메모리의 내용을 ret에 추가합니다.
		return ret


	"""
	calc[index] = value 꼴로 캐시를 저장하고 싶을 떄, __setitem__(self, index(==key), value) 메서드를 이용해 구현합니다.
	key는 0이상 5미만의 정수, value는 int 또는 float인지 확인하고, 이에 해당할 때만 cache에 저장힙니다.
	여기서는 type 함수를 이용해서 변수형을 확입합니다.
	"""

	def __setitem__(self, key, value):
		if str(type(key)) != "<class 'int'>" or key < 0 or key >= 5: # key가 0이상 5미만의 정수가 아니면 종료합니다.
			return

		if str(type(value)) != "<class 'int'>" and str(type(value)) != "<class 'float'>": # value가 int나 float가 아니면 종료합니다.
			return

		self.cache[key] = value # cache에 값을 저장합니다.