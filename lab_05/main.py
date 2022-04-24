from HopfildNetwork import HopfildNetwork

# Асинхронный режим
# Запоминаемые образы : 3 5 7

if __name__ == '__main__':
    hopfildNetwork = HopfildNetwork([3, 5, 7])
    # print(hopfildNetwork.getMatrixArrayStr())
    print(hopfildNetwork)
    vector = [1, 1, 1, -1, -1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 1]
    print(vector)
    print(hopfildNetwork.getOutputVector(vector))
