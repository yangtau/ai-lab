import sys
import math
import numpy as np
from collections import defaultdict, Counter
import plot_tree


def compute_entropy(data, split_point):
    left_cnt = defaultdict(lambda: 0)
    right_cnt = defaultdict(lambda: 0)
    num_left = 0
    num_right = 0
    for x, y in data:
        if x < split_point:
            left_cnt[y] += 1
            num_left += 1
        else:
            right_cnt[y] += 1
            num_right += 1
    log = math.log2
    return (
        sum(-v/num_left*log(v/num_left)
            for v in left_cnt.values()) * num_left +
        sum(-v/num_right*log(v/num_right)
            for v in right_cnt.values()) * num_right
    ) / len(data)


'''
def compute_entropy(data, split_point):
    left = 0
    n = len(data)
    cnt = defaultdict(lambda: 0)
    for x, y in data:
        if x < split_point:
            left += 1
            cnt[(0, y)] += 1
        else:
            cnt[(1, y)] += 1
    pxl = left / n  # P(x<split_point)
    pxr = 1 - pxl  # P(x>=split_point)
    # H(X|Y) = sum(P(x, y) * log(P(x)/P(x, y)))
    return sum(c/n * math.log2((pxl if i == 0 else pxr)*n/c)
               for (i, y), c in cnt.items())
'''


def get_best_split(xs, ys):
    '''return the entropy and the best splitting point'''
    if len(xs) == 0:
        return float('+inf'), 0
    data = list(zip(xs, ys))
    data.sort()
    split_points = set((data[i][0]+data[i+1][0])/2 for i in range(len(xs)-1))
    entropies = [(compute_entropy(data, spl), spl)
                 for spl in split_points]
    return min(entropies)


def split_data(data_set, labels):
    '''return idx, split_point, (left, left_labels), (right, right_labels)
       idx: the index of attribute in data_set which has the minimum entropy
       split_point: the split point of the chosen attribute
    '''
    entropies = [get_best_split(xs, labels) for xs in data_set]
   #  min_entropy = min(entropies)
    min_entropy = entropies[0]
    for e in entropies:
        if e[0] < min_entropy[0]:
            min_entropy = e
    idx = entropies.index(min_entropy)
    split_point = min_entropy[1]
    left_idxs = np.where(data_set[idx] < split_point)[0]
    right_idxs = np.where(data_set[idx] >= split_point)[0]
    left = data_set[..., left_idxs]
    left_labels = labels[left_idxs]
    right_labels = labels[right_idxs]
    right = data_set[..., right_idxs]
    return idx, split_point, (left, left_labels), (right, right_labels)


def create_tree(data_set, labels, max_height):
    '''return the decision tree
       date structure of the tree: (Node, left, right)
       If the node is a leaf, then it is the value of the label,
       else it is (idx, split_point).
    '''
    if len(set(labels)) == 1:
        # return if there is only one label left
        return (labels[0], None, None)
    if max_height == 1:
        # reach the max height
        cnt = Counter(labels)
        return (cnt.most_common()[0][0], None, None)
    idx, split_point, left, right = split_data(data_set, labels)
    return (idx, split_point), create_tree(*left, max_height-1),\
        create_tree(*right, max_height-1)


def test_data(data, decision_tree):
    ''' return the label of data
        data: [x1, x2, ..., xn]
    '''
    cur, left, right = decision_tree
    if left is None and right is None:
        return cur
    idx, split_point = cur
    return test_data(data, left) if data[idx] < split_point \
        else test_data(data, right)


def test(data_set, labels, tree):
    data_set = data_set.T
    n = len(data_set)
    cnt = 0
    for i in range(n):
        y = test_data(data_set[i], tree)
        print(data_set[i], labels[i], 'res: {}'.format(y))
        if y == labels[i]:
            cnt += 1
    return cnt / n


def load_data(filename):
    '''return data_set, labels'''
    with open(filename) as f:
        lines = f.readlines()[1:-1]
    lines = [l.replace('\t', ' ').strip().split(' ') for l in lines]
    data = np.array(list(list(map(float, l)) for l in lines))
    return data[..., :-1].T, data[..., -1]


def main(traindata_file, testdata_file):
    data_set, labels = load_data(traindata_file)
    tree = create_tree(data_set, labels, 8)
    test_data_set, test_labes = load_data(testdata_file)
    accrucy = test(test_data_set, test_labes, tree)
    print(accrucy)
    plot_tree.createPlot(tree)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: {} traindata testdata'.format(sys.argv[0]))
    main(sys.argv[1], sys.argv[2])
