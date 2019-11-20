#!/usr/bin/env python3
import random
import queue
from collections import defaultdict
import math
import sys


'''
state
((2, 2),
(1, 2, 3,
 4, 5, 6,
 7, 8, 0))
0: empty cell
'''


class PQEntry:
    def __init__(self, state, g, h, prev):
        self.state = state
        self.g = g
        self.h = h
        self.f = g + h
        self.prev = prev

    def __eq__(self, other):
        return self.f == other.f

    def __lt__(self, other):
        return self.f < other.f


def search(init_state, dest_check, transform, h_estimate):
    pq = queue.PriorityQueue()
    pq.put(PQEntry(init_state, 0, h_estimate(init_state), None))
    # visited {state: g}
    visited = defaultdict(lambda: math.inf)
    visited[init_state] = 0
    while not pq.empty():
        front = pq.get()
        if dest_check(front.state):
            return front
        trans = transform(front.state)
        for g_inc, s in trans:
            g = g_inc + front.g
            if g < visited[s]:
                visited[s] = g
                h = h_estimate(s)
                pq.put(PQEntry(s, g, h, front))
    # No solution found
    return None


def transform(state):
    (ex, ey), layout = state
    layout = list(layout)
    exchange = ((ex+1, ey), (ex-1, ey), (ex, ey+1), (ex, ey-1))

    def fn(x, y):
        t = layout.copy()
        t[x*3+y], t[ex*3+ey] = t[ex*3+ey], t[x*3+y]
        return tuple(t)

    return [(1, ((x, y), fn(x, y))) for x, y in exchange
            if x not in (-1, 3) and y not in (-1, 3)]


def dest_check(state):
    _, layout = state
    return layout == (1, 2, 3,
                      8, 0, 4,
                      7, 6, 5)


def hamilton_dist_sum(state):
    des_position = ((1, 1), (0, 0), (0, 1),
                    (0, 2), (1, 2), (2, 2),
                    (2, 1), (2, 0), (1, 0))
    _, layout = state

    def fn(i):
        m, n = i // 3, i % 3
        x, y = des_position[layout[i]]
        return abs(x-m)+abs(y-n)
    return sum(fn(i) for i in range(9))


def wrong_position_cnt(state):
    right_layout = (1, 2, 3,
                    8, 0, 4,
                    7, 6, 5)
    _, layout = state
    return sum(1 for i, j in zip(layout, right_layout) if i != j)


def print_sl(res):
    if res.prev:
        print_sl(res.prev)
    print(res.state[1][0:3])
    print(res.state[1][3:6])
    print(res.state[1][6:9])
    print()


def solve(start, h_estimate):
    mark = [0]*9
    for i in range(9):
        if start[i] == 0:
            j = i
        try:
            mark[start[i]] = 1
        except Exception:
            print('Invalid input')
            exit()
    for m in mark:
        if m != 1:
            print('Invalid input')
            exit()

    res = search(((j//3, j % 3), start), dest_check, transform, h_estimate)
    print("==========")
    if not res:
        print(start[0:3])
        print(start[3:6])
        print(start[6:9])
        print('No solution found')
    else:
        print_sl(res)
    print("==========")


if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == 'wrong':
        print('wrong')
        h_estimate = wrong_position_cnt
    else:
        h_estimate = hamilton_dist_sum

    lines = []
    try:
        for l in sys.stdin:
            lines.append(l)
    except StopIteration:
        pass
    ipt = ''.join(lines)
    ipt = ipt.replace('\n', '')
    ipt = ipt.replace(' ', '')
    ipt = ipt.replace('\t', '')
    if len(ipt) % 9 != 0:
        print('Invalid input')
        exit()
    s = 0
    while s < len(ipt):
        solve(tuple(map(int, ipt[s:s+9])), h_estimate)
        s += 9
