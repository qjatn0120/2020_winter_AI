from copy import copy # keys, values 함수에서 사용합니다.

class Dictionary:

	def __init__(self):
		self.key_list = [] # 딕셔너리의 key를 저장하는 리스트를 정의합니다.
		self.value_list = [] # 딕셔너리의 value를 저장하는 리스트를 정의합니다.

	# {(0번째 key : 0번째 value), (1번째 key : 1번째 value), ...} 의 형식으로 출력합니다.
	def __str__(self):
		ret = "{" # 출력할 문자열을 ret에 저장힙니다.
		for x in range(len(self.key_list)):
			ret += "({} : {})".format(self.key_list[x], self.value_list[x]) # key : value 포맷에 맞게 ret에 추가힙니다.
			if x != len(self.key_list) - 1: # 마지막 원소가 아니라면 컴마(,) 를 추가합니다.
				ret += ", "
		ret += "}"
		return ret

	"""
	딕셔너리는 dic[“one”] = 1과 같은 형식으로 값을 추가합니다. (key는 “one”, value는 “1”이 됩니다.)
	만약 “one” : 1 이 존재하는 데, dic[“one”] = 2를 한다면,
	새로운 key : value가 추가되는 것이 아니라, “one” : 2 로 해당 key의 value를 수정해야 합니다.
	__setitem__ 메서드를 이용해 구현합니다.
	list.index(x) 함수는 x가 저장되어 있는 index를 반환합니다. 만약 list에 x가 존재하지 않는다면 에러가 발생합니다.
	list.index(x)의 에러를 막기 위해 x in list 를 이용하여 list에 x가 있는 지 확인할 수 있습니다.
	"""
	def __setitem__(self, key, value):
		if key in self.key_list: # 만약 key가 key_list에 존재한다면
			idx = self.key_list.index(key) # key가 저장된 index를 찾은 후
			self.value_list[idx] = value # value 값만 수정합니다.
		else: # 만약 key가 key_list에 존재하지 않는다면
			# key와 value를 list에 추가합니다.
			self.key_list.append(key)
			self.value_list.append(value)

	# dic[key]로 value에 접근합니다. __getitem__으로 구현합니다.
	def __getitem__(self, key):
		idx = self.key_list.index(key) # key가 저장된 index를 찾은 후
		return self.value_list[idx] # value를 반환합니다.

	# dic.pop(key) 함수로 (key, value)를 삭제합니다.
	def pop(self, key):
		idx = self.key_list.index(key) # key가 저장된 index를 찾은 후
		ret = self.value_list[idx] # 반환할 값을 미리 저장합니다.
		del self.key_list[idx] # key를 삭제합니다.
		del self.value_list[idx] # value를 삭제합니다.
		return ret # value를 반환합니다.

	# dic.keys() 함수는 key list를 반환합니다.
	def keys(self):
		return copy(self.key_list)

	# dic.values() 함수는 value list를 반환합니다.
	def values(self):
		return copy(self.value_list)

	# dic.items() 함수는 (key, value) 튜플 리스트를 반환합니다.
	def items(self):
		ret = [] # 튜플 리스트
		for x in range(len(self.key_list)):
			ret.append((self.key_list[x], self.value_list[x]))# 튜플 (key, value)를 리스트에 추가합니다.
		return ret

	# dic.clear() 함수는 key list와 value list를 비웁니다.
	def clear(self):
		del self.key_list[:] # key_list의 모든 원소를 삭제합니다.
		del self.value_list[:] # value_list의 모든 원소를 삭제합니다.
