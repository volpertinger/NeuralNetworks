from HopfildNetwork import HopfildNetwork
from numpy import array

# Асинхронный режим
# Запоминаемые образы : 3 5 7

# 3 = [1, 1, 1, 1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1, 1, 1]
# 5 = [1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1, 1, 1]
# 7 = [1, 1, 1, 1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1, 1]

if __name__ == '__main__':
    hopfildNetwork = HopfildNetwork([3, 5, 7])
    print(hopfildNetwork)
    vectors_initial = [[1, 1, 1, 1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1, 1, 1],
                       [1, 1, 1, 1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1, 1]]
    vectors_corrupted = [[-1, 1, 1, 1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1, 1, 1],
                         [1, -1, -1, 1, -1, -1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1, 1, 1],
                         [1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1, 1, -1, 1, -1, 1, 1, 1, -1, -1, 1, 1, 1, 1, 1, 1],
                         [1, 1, 1, 1, 1, 1, -1, -1, 1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1],
                         [1, 1, 1, 1, -1, -1, -1, 1, -1, -1, -1, 1, 1, -1, -1, 1, -1, -1, -1, 1],
                         [1, 1, 1, 1, -1, -1, -1, 1, -1, -1, -1, 1, -1, 1, 1, 1, -1, -1, -1, 1],
                         [1, 1, 1, 1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1, -1, 1, 1, -1, 1]]
    # проверка на полностью верных данных
    print('Right vectors\n')
    for vector in vectors_initial:
        print('input:  ', array(vector))
        print(hopfildNetwork.getStrMatrixFromVector(vector))
        output = hopfildNetwork.getOutputVector(vector)
        print('output: ', output)
        print(hopfildNetwork.getStrMatrixFromVector(output))
        print()
    # проверка частично искаженных векторов
    print('Corrupted vectors\n')
    for vector in vectors_corrupted:
        print('input:  ', array(vector))
        print(hopfildNetwork.getStrMatrixFromVector(vector))
        output = hopfildNetwork.getOutputVector(vector)
        print('output: ', output)
        print(hopfildNetwork.getStrMatrixFromVector(output))
        print()
