# # 최대 길이를 제외한 그 다음 길이부터 생각해서 잘랐을 때 길이가 손님이 원하는 길이보다 작으면 다음 원소 비교
# n, m = map(int, input().split()) 
# array = list(map(int, input().split())) 

# # 시작점과 끝점
# start = 0
# end = max(array)

# result = 0
# while(start <= end):
#     total = 0
#     mid = (start + end) // 2
#     for x in array:
#         if x > mid:
#             total += x - mid # 잘랐을 때 떡의 양 계산
#     # 떡의 길이가 부족한 경우, 왼쪽 부분 탐색
#     if total < m:
#         end = mid - 1
#     else: # 떡의 양이 충분한 경우
#         result = mid # 현재 만족한 상태에서의 값을 저장
#         start = mid + 1 # 최댓값이 있는지 오른쪽 부분 탐색

# print(result)
# from bisect import bisect_left, bisect_right

# def count_by_range(array, left_value, right_value):
#     right_index = bisect_right(array, right_value)
#     left_index = bisect_left(array, left_value)
#     return right_index - left_index

# n, x = map(int, input().split())
# array = list(map(int, input().split()))

# count = count_by_range(array, x, x)
# if count == 0:
#     print(-1)
# else:
#     print(count)

# 피보나치 함수를 재귀함수로 구현
# def fibo(x):
#     if x == 1 or x == 2:
#         return 1
#     return fibo(x-1) + fibo(x-2)

# print(fibo(50))

# 탑다운 방식의 피보나치 수열
# 한 번 계산된 결과를 메모이제이션 하기 위한 리스트 초기화
# d = [0] * 100
# def fibo(x):
#     if x == 1 or x == 2: # 종료 조건
#         return 1
#     if d[x] != 0: # 이전에 계산된 적이 있으면 그대로 반환
#         return d[x]
#     d[x] = fibo(x-1) + fibo(x-2) # 계산된 적이 없으면 피보나치 결과 반환
#     return d[x]

# print(fibo(99))

# 보텀업 방식의 피보나치 수열
d = [0] * 100
d[1] = 1
d[2] = 1
n = 99

# 재귀함수가 아닌 반복문이 사용됨
# for i in range(3, n+1):
#     d[i] = d[i-1] + d[i-2]

# print(d[n])

# 개미 전사
n = int(input())
array = list(map(int, input().split()))
d = [0] * 100 # 문제에서 최대 100이라고 제시했기 때문
# 다이나믹 프로그래밍 진행(보텀업 방식)
d[0] = array[0]
d[1] = max(array[0], array[1])
for i in range(2, n):
    d[i] = max(d[i-1], d[i-2] + array[i])

print(d[n-1])