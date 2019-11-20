import random
import sys


def get_random_state(num: int):
    state = [1, 2, 3,
             8, 0, 4,
             7, 6, 5]
    position = 1, 1

    def random_position(pos):
        move = [(0, 1), (0, -1), (-1, 0), (1, 0)]
        i = random.randint(0, 3)
        x, y = pos[0]+move[i][0], pos[1]+move[i][1]
        while x < 0 or y < 0 or x == 3 or y == 3:
            i = random.randint(0, 3)
            x, y = pos[0]+move[i][0], pos[1]+move[i][1]
        return x, y
    for _ in range(num):
        new_pos = random_position(position)
        new_pos_idx = new_pos[0]*3+new_pos[1]
        pos_idx = position[0]*3+position[1]
        state[pos_idx], state[new_pos_idx] = state[new_pos_idx], state[pos_idx]
        position = new_pos
    return state


if __name__ == '__main__':
    n = int(sys.argv[1])
    for _ in range(n):
        i = random.randint(5, 100)
        state = get_random_state(i)
        print('%d %d %d\n%d %d %d\n%d %d %d\n' % tuple(state))

