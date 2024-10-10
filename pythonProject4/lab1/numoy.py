import numpy as np
import matplotlib.pyplot as plt

def sum_prod(X,V): #1
    S = 0
    for X, V in zip(X, V):
        S += np.dot(X, V)
    return np.ndarray.tolist(S)
def test_sum_prod():
    assert sum_prod([[[1,2,3],[4,5,6],[7,8,9]],[[4,5,6],[4,5,6],[7,8,9]],[[4,5,6],[4,5,6],[7,8,9]]],[1,2,3]) == [[21, 27, 33], [24, 30, 36], [42, 48, 54]]
    assert sum_prod([[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[4, 5, 6], [4, 8, 8], [7, 8, 9]], [[4, 5, 6], [4, 5, 6], [7, 8, 9]]],[4, 5, 3]) == [[36, 48, 60], [48, 75, 82], [84, 96, 108]]
    assert sum_prod([[1,2,3],[4,5,6],[7,8,9]],[1,2,3]) == [30, 36, 42]

test_sum_prod()


print(sum_prod([[1,2,3],[4,5,6],[7,8,9]],[1,2,3]))
def binarize(M, threshold): # 2
    A = []
    B = []
    for i in range(len(M)):
        A = []
        for j in range(len(M[0])):
            if M[i][j] >= threshold:
                A.append(1)
            else:
                A.append(0)
        B.append(A)
    return B
def test_binarize():
    assert binarize([[1,2,3],[4,5,6],[7,8,9]], 0.5) == [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    assert binarize([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 2) == [[0, 1, 1], [1, 1, 1], [1, 1, 1]]
    assert binarize([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 7) == [[0, 0, 0], [0, 0, 0], [1, 1, 1]]
test_binarize()

def unique_rows(mat):
    rows = []
    for row in mat:
        unique_row = np.unique(row)
        rows.append(np.ndarray.tolist(unique_row))
    return rows
def unique_columns(mat):  #3
    column = []
    for columns in mat.T:
        col = np.unique(columns)
        column.append(np.ndarray.tolist(col))
    return column
def test_unique():
    assert unique_columns(np.array([[1,2,3],[4,5,6],[7,8,9]])) == [[1, 4, 7], [2, 5, 8], [3, 6, 9]]
    assert unique_columns(np.array([[1,3,3],[4,6,6],[9,8,9]])) == [[1, 4, 9], [3, 6, 8], [3, 6, 9]]
    assert unique_rows(np.array([[1,2,3],[4,5,6],[7,8,9]])) == [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    assert unique_rows(np.array([[1,3,3],[4,6,6],[9,8,9]])) == [[1, 3], [4, 6], [8, 9]]
test_unique()

def chess(m, n, a, b): #5
    A = []
    for i in range(m):
        B = []
        for j in range(n):
            if i % 2 == 0 and j % 2 == 0:
                B.append(a)
            if i % 2 == 0 and j % 2 != 0:
                B.append(b)
            if i % 2 != 0 and j % 2 == 0:
                B.append(b)
            if i % 2 != 0 and j % 2 != 0:
                B.append(a)
        A.append(B)
    return A
def chess_test():
    assert chess(3,3,1,0) == [[1, 0, 1], [0, 1, 0], [1, 0, 1]]
    assert chess(4,4,1,0) == [[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]]
    assert chess(2,2,1,2) == [[1, 2], [2, 1]]
chess_test()
#print(chess(2,2,1,2))
#M = [[1, 2, 3], [4, 5, 6, 4], [7, 8, 9, 7]]
#print(unique_columns(M))
#A = np.arange(6).reshape(3,2)
#print(unique_columns(A))