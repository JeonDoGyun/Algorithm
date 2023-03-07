# # 이진 탐색 소스 코드
# def binary_search(array, target, start, end):
#     if start > end:
#         return None
#     mid = (start + end) / 2
#     # 찾은 경우 중간인덱스 반환
#     if array[mid] == target:
#         return mid
#     # 중간인덱스의 값이 타겟보다 클 경우 -> 왼쪽까지의 값을 다시 이진 탐색
#     elif array[mid] > target:
#         return binary_search(array, target, start, mid - 1)
#     # 중간인덱스의 값이 타겟보다 작을 경우 -> 중간인덱스의 오른쪽을 시작점으로 다시 이진 탐색
#     else:
#         return binary_search(array, target, mid + 1, end)
    
# n, target = list(map(int, input().split()))
# array = list(map(int, input().split()))

# result = binary_search(array, target, 0, n-1)
# if result == None:
#     print("원소가 존재하지 않습니다.")
# else:
#     print(result + 1)

# 파이썬 이진탐색 라이브러리
from bisect import bisect_left, bisect_right

a = [1, 2, 4, 4, 8]
x = 4

print(bisect_left(a, x))
print(bisect_right(a, x))