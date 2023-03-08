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
from bisect import bisect_left, bisect_right

def count_by_range(array, left_value, right_value):
    right_index = bisect_right(array, right_value)
    left_index = bisect_left(array, left_value)
    return right_index - left_index

n, x = map(int, input().split())
array = list(map(int, input().split()))

count = count_by_range(array, x, x)
if count == 0:
    print(-1)
else:
    print(count)