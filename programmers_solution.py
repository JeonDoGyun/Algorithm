def solution(a, b, n):
    answer = 0
    bottle = 0
    
    while True:
        bottle = (n // a) * b # 마트에 주고 받은 병의 수
        answer += bottle
        n = bottle + n % a
        print(n)
        if n < a:
            break
    return answer

print(solution(4, 3, 15))