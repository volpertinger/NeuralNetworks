from NeuralNetworkRBS import NeuralNetworkRBS

if __name__ == '__main__':
    # (x1 + x2)x3 + x4
    function = [0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1]

    # (!(x1x2)x3x4
    # function = [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]
    print("Simple function, all vectors")
    boolNeuron = NeuralNetworkRBS(function)
    boolNeuron.teach()
    print(boolNeuron.getLogStr())
    print()

    print("Not Simple function, all vectors")
    boolNeuron = NeuralNetworkRBS(function, False)
    boolNeuron.teach()
    print(boolNeuron.getLogStr())
    print()

    print("Simple function, minimum vectors")
    boolNeuron = NeuralNetworkRBS(function)
    minIndexes = boolNeuron.getMinTeachIndexes()
    boolNeuron.setTeachIndexes(minIndexes)
    boolNeuron.teach()
    print(minIndexes)
    print(boolNeuron.getLogStr())
    print()

    print("Not Simple function, minimum vectors")
    boolNeuron = NeuralNetworkRBS(function, False)
    minIndexes = boolNeuron.getMinTeachIndexes()
    boolNeuron.setTeachIndexes(minIndexes)
    boolNeuron.teach()
    print(minIndexes)
    print(boolNeuron.getLogStr())
