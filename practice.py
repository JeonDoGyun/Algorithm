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
# d = [0] * 100
# d[1] = 1
# d[2] = 1
# n = 99

# 재귀함수가 아닌 반복문이 사용됨
# for i in range(3, n+1):
#     d[i] = d[i-1] + d[i-2]

# print(d[n])

# 개미 전사
# n = int(input())
# array = list(map(int, input().split()))
# d = [0] * 100 # 문제에서 최대 100이라고 제시했기 때문
# # 다이나믹 프로그래밍 진행(보텀업 방식)
# d[0] = array[0]
# d[1] = max(array[0], array[1])
# for i in range(2, n):
#     d[i] = max(d[i-1], d[i-2] + array[i])

# print(d[n-1])

# 1로 만들기
# 지금 숫자에서 할 수 있는 경우 탐색하고 실행, 그 뒤의 경우의 수를 보고 데이터 저장, 데이터 중 최솟값 출력
# 19 -> 18 -> 9 -> 3 -> 3
# 19 -> 18 -> 6 -> 2 -> 1

# x = int(input())
# d = [0] * 30001

# for i in range(2, x+1):
#     # 현재의 수에서 1을 뺀 경우
#     d[i] = d[i-1] + 1
#     # 현재의 수가 2로 나누어 떨어지는 경우
#     if i % 2 == 0:
#         d[i] = min(d[i], d[i//2] + 1) # 1로 뺀 경우와 비교하여 더 작은 값으로 변경
#     # 현재의 수가 3으로 나누어 떨어지는 경우
#     if i % 3 == 0:
#         d[i] = min(d[i], d[i//3] + 1)
#     # 현재의 수가 5로 나누어 떨어지는 경우
#     if i % 5 == 0:
#         d[i] = min(d[i], d[i//5] + 1)

# print(d[x])

# 효율적인 화폐 구성
# i번째 구하려면 bills의 최대로 빼고 남은 것의 다시 a16 = a13 + 3
# n, m = map(int, input().split()) # 2, 16
# array = []
# for i in range(n):
#     array.append(int(input()))

# d = [10001] * (m + 1) # m은 10000까지이기 때문에 계산될 수 없는 값인 10001을 임의로 넣어준 것
# d[0] = 0
# for i in range(n):
#     for j in range(array[i], m + 1):
#         # 현재 금액에서 확인중인 화폐 단위를 뺀 금액을 만들 수 있으면 갱신 ex) 4 => 3을 봤을 때 1을 못만드니 패스, 2를 봤을 때 2를 만들 수 있으니 갱신
#         if d[j - array[i]] != 10001: 
#             d[j] = min(d[j], d[j - array[i]] + 1)

# if d[m] == 10001:
#     print(-1)
# else:
#     print(d[m])

# 금광 문제
# for tc in range(int(input())):
#     n, m = map(int, input().split())
#     array = list(map(int, input().split()))

#     dp = []
#     index = 0
#     for i in range(n):
#         dp.append(array[index:index + m]) # 1차원 데이터를 슬라이싱을 통해 2차원 데이터로 변환
#         index += m
    
#     for j in range(1, m):
#         for i in range(n):
#             # 왼쪽 위에서 오는 경우
#             if i == 0: left_up = 0 # index 벗어난 경우
#             else: left_up = dp[i-1][j-1]
#             # 왼쪽 아래에서 오는 경우
#             if i == n-1: left_down = 0 # index 벗어난 경우
#             else: left_down = dp[i+1][j-1]
#             # 왼쪽에서 오는 경우
#             left = dp[i][j-1]
#             # 금 개수 변경
#             dp[i][j] = dp[i][j] + max(left, left_down, left_up)
#     result = 0
#     for i in range(n):
#         result = max(result, dp[i][m-1]) # 가장 오른쪽 열에 있는 값 중 최댓값을 고르면 정답
#     print(result)

# 병사 배치
# n = int(input())
# array = list(int, input().split())
# array.reverse()

# dp = [1] * n
# for i in range(1, n):
#     for j in range(0, i):
#         if array[j] < array[i]:
#             dp[i] = max(dp[i], dp[j]+1)

# print(n-max(dp))

# # 다익스트라 최단경로 알고리즘
# import sys
# input = sys.stdin.readline
# INF = int(1e9)

# n, m = map(int, input().split())
# start = int(input())
# # 각 노드에 연결되어 있는 노드에 대한 정보를 담는 리스트 만들기
# graph = [[] for i in range(n + 1)]
# visited = [False] * (n + 1)
# # 최단 거리 테이블을 모두 무한으로 초기화
# distance = [INF] * (n + 1)

# # 모든 간선 정보를 입력받기
# for _ in range(m):
#     a, b, c = map(int, input().split())
#     graph[a].append(b, c) # a번 노드에서 b번 노드로 가는 비용이 c

# # 방문하지 않은 노드 중에서, 가장 최단 거리가 짧은 노드의 번호를 반환
# def get_smallest_node():
#     min_value = INF
#     index = 0 # 가장 최단 거리가 짧은 노드(인덱스)
#     for i in range(1, n+1):
#         if distance[i] < min_value and not visited[i]:
#             min_value = distance[i]
#             index = 1
#     return index

# def dijkstra(start):
#     # 시작 노드에 대하여 초기화
#     distance[start] = 0
#     visited[start] = True
#     for j in graph[start]:
#         distance[j[0]] = j[1]
    
#     # 시작 노드를 제외한 전체 n-1개의 노드에 대해 반복
#     for i in range(n-1):
#         # 현재 최단 거리가 가장 짧은 노드를 꺼내서, 방문 처리
#         now = get_smallest_node()
#         visited[now] = True
#         # 현재 노드와 연결된 다른 노드 확인
#         for j in graph[now]:
#             cost = distance[now] + j[1]
#             # 현재 노드를 거쳐서 다른 노드로 이동하는 거리가 더 짧은 경우
#             if cost < distance[j[0]]:
#                 distance[j[0]] = cost

# dijkstra(start)

# # 모든 노드로 가기 위한 최단 거리 출력
# for i in range(1, n+1):
#     if distance[i] == INF:
#         print("Infinity")
#     else:
#         print(distance[i])

# 최소 힙
# import heapq

# def heapsort(iterable):
#     h = []
#     result = []
#     # 모든 원소를 차례대로 힙에 삽입
#     for value in iterable:
#         heapq.heappush(h, value)
#     # 힙에 삽입된 모든 원소를 차례대로 꺼내어 담기
#     for i in range(len(h)):
#         result.append(heapq.heappop(h))
#     return result

# result = heapsort([1, 3, 5, 7, 9, 2, 4, 6, 8, 0])
# print(result)

# 개선된 다익스트라 알고리즘(heap)
import heapq
import sys

input = sys.stdin.readline
INF = int(1e9)

n, m = map(int, input().split())
start = int(input())
graph = [[] for i in range(n+1)]
distance = [INF] * (n+1)

for _ in range(m):
    a, b, c = map(int, input().split())
    graph[a].append((b, c))

def dijkstra(start):
    q = []
    heapq.heappush(q, (0, start)) # (거리, 시작지점)
    distance[start] = 0

    while q: # 큐가 비어있지 않다면 반복
        dist, now = heapq.heappop(q)
        if distance[now] < dist: # 현재 노드의 거리가 저장된 값보다 더 크면 = 현재의 노드가 이미 처리된 적이 있는 노드라면 무시
            continue
        for i in graph[now]:
            cost = dist + i[1]
            if cost < distance[i[0]]:
                distance[i[0]] = cost
                heapq.heappush(q, (cost, i[0]))

dijkstra(start)

for i in range(1, n+1):
    if distance[i] == INF:
        print("Infinity")
    else:
        print(distance[i])