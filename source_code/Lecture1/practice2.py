"""
연습 문제2

세 분반의 성적이 주어질 때, 리스트를 매개변수로 받아 평균값(average), 중앙값(median), 최대값(maximum), 최소값(minimum)을 계산하는 함수를 각각 만드세요.
리스트와 분반 번호가 입력으로 주어질 때, 위에서 만든 함수를 이용하여 분반 성적 정보를 출력하는 함수(print_info)를 만드세요

출력 형식
(분반 번호)분반
학생 수 : (학생 수)명
평균 점수 : (평균 점수)점
중앙값 : (중앙값)점
최고 점수 : (최고 점수)점, 최저 점수 : (최저 점수) 점

평균 점수는 소수점 2자리까지 반올림하여 출력합니다.
중앙값은 소수점 1자리까지 출력합니다.
중앙값은 학생 수 N이 홀수인 경우, (N + 1) / 2번째 학생의 점수, N이 짝수인 경우, N / 2번째 학생과 (N / 2) + 1번째 학생의 성적의 평균으로 정의합니다.

1번째 줄에는 1분반 학생의 성적, 2번째 줄에는 2분반 학생의 성적, 3번째 줄에는 3분반 학생의 성적이 주어집니다.
세 분반의 성적을 입력으로 받아 위 출력형식에 맞게 분반 성적 정보를 출력하면 됩니다.

세 분반의 성적을 입력으로 받아 위 출력형식에 맞게 분반 성적 정보를 출력하면 됩니다.
(이처럼 줄 단위가 아니라 띄어쓰기 단위로 입력이 주어지는 문제는 split 함수를 사용해서 문장 전체를 입력받고,
띄어쓰기 단위로 나눠 리스트에 저장할 수 있습니다.)

Example input
98 75 68 45 78
67 95 74 82 46 48
91 87 82 76 43 66 71

Example output
1분반
학생 수 : 5명
평균 점수 : 72.80점
중앙값 : 75.00점
최고 점수 : 98점, 최저 점수 : 45점

2분반
학생 수 : 6명
평균 점수 : 68.67점
중앙값 : 70.50점
최고 점수 : 95점, 최저 점수 : 46점

3분반
학생 수 : 7명
평균 점수 : 73.71점
중앙값 : 76.00점
최고 점수 : 91점, 최저 점수 : 43점
"""

def average(List):
	Sum = 0 # 점수 총합
	for score in List:
		Sum += score
	return Sum / len(List)

def median(List):
	ret = 0 # 중앙값
	N = len(List) # 학생 수
	# List index는 0부터 시작하므로, 1을 빼주는 것을 잊지 맙시다.
	if N % 2 == 0: # 학생 수가 짝수인 경우
		ret = (List[N // 2 - 1] + List[N // 2]) / 2 # 중앙값 = (N/2번째 학생 점수 + N/2+1번째 학생 점수) / 2
	else : # 학생 수가 홀수인 경우
		ret = List[(N - 1) // 2] # 중앙값 = (N + 1) / 2번째 학생 점수는

	return ret

def maximum(List):
	# List[-x]는 뒤에서 x번째 원소를 의미합니다.
	return List[-1]

def minimum(List):
	return List[0]

def print_info(idx, List):
	# 먼저 list의 원소가 string형이므로 int로 바꿔줍니다.
	for i in range(len(List)): # range(0, x) == range(x)
		List[i] = int(List[i])
	# 리스트의 원소를 정렬해줍니다.
	List.sort()
	# 형식에 맞게 분반 정보를 출력합니다.
	print("{}분반".format(idx))
	print("학생 수 : {}명".format(len(List)))
	print("평균 점수 : {:.2f}점".format(average(List)))
	print("중앙값 : {:.2f}점".format(median(List)))
	print("최고 점수 : {}점, 최저 점수 : {}점\n".format(maximum(List), minimum(List)))

# 세 분반의 학생 성적을 입력받고 띄어쓰기 단위로 분리하여 리스트에 저장합니다.
List1 = input().split(' ')
List2 = input().split(' ')
List3 = input().split(' ')

print_info(1, List1)
print_info(2, List2)
print_info(3, List3)