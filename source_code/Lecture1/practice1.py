"""
연습문제 1

학생 세 명의 성적(0≤𝑥≤100)이 한 줄에 하나씩 총 세 줄에 걸쳐 주어집니다. 이 때, 세 학생의 평균 성적을 소수점 2자리까지 반올림해서 출력해봅시다.

Average score : (평균성적) 형태로 출력해봅시다.

Example input
96
87
91

Example output
Average score : 91.33
"""

score1 = int(input())
score2 = int(input())
score3 = int(input())
avg_score = (score1 + score2 + score3) / 3
print("Average score : {:.2f}".format(avg_score))