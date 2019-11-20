#!/usr/bin/env python3

import queue
import sys

'''
state (ML, CL, BL)
`ML`: the number of missionaries on the left bank.
`CL`: the number of cannibals on the left bank.
`BL`: the number of boats on the left bank.
'''


def search(init_state, dest_check: callable, transform: callable):
    que = queue.Queue()
    # visited = set()

    # ((state), prev_state)
    que.put((init_state, None))
    # visited.add(init_state)
    res = []
    while not que.empty():
        front = que.get()
        if dest_check(front[0]):
            # return front
            res.append(front)
        trans = transform(front[0])
        for t in trans:
            _, prev = front
            while prev != None and prev[0] != t:
                _, prev = prev
            if prev == None:
                que.put((t, front))
            '''
            if t not in visited:
                que.put((t, front))
                visited.add(t)
            '''
    return res
    # return None  # No solution


def transform_func(num_miss: int, num_cann: int, boat_capacity: int):
    # check if the state is valid
    def check(ml, cl):
        mr, cr = num_miss - ml, num_cann - cl
        return ml >= 0 and cl >= 0 and \
            mr >= 0 and cr >= 0 and \
            (ml == 0 or ml >= cl) and \
            (mr == 0 or mr >= cr)

    def fn(state):
        ml, cl, bl = state
        if bl == 1:
            res = [(ml-j, cl-(i-j), 0) for i in range(1, boat_capacity+1)
                   for j in range(0, i+1) if check(ml-j, cl-(i-j))]
        else:
            res = [(ml+j, cl+(i-j), 1) for i in range(1, boat_capacity+1)
                   for j in range(0, i+1) if check(ml+j, cl+(i-j))]
        return res
    return fn


def dest_check(state):
    ml, cl, _ = state
    return ml == 0 and cl == 0


def print_solu(res):
    if not res:
        return
    print_solu(res[1])
    print(res[0])


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Usage: program missionaries cannibals capacity')
        exit(-1)

    num_miss, num_cann, capacity = int(
        sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    transf = transform_func(num_miss, num_cann, capacity)

    res = search((num_miss, num_cann, 1), dest_check, transf)
    if not res:
        print('No solution found.')
    else:
        print(len(res))
        for r in res:
            print_solu(r)
            print()
