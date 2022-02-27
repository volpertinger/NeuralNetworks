from BoolNeuron import BoolNeuron

if __name__ == '__main__':
    '''''
    
    (X1+X2)X3+X4
    
    F = 0 1 0 1 0 1 1 1 0 1 1 1 0 1 1 1
    '''''

    function = [0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1]

    boolNeuron = BoolNeuron(function)
    boolNeuron.teach()
    print(boolNeuron.getLogStr())
    print()

    boolNeuron = BoolNeuron(function, False)
    boolNeuron.teach()
    print(boolNeuron.getLogStr())
    print()

    boolNeuron = BoolNeuron(function)
    minIndexes = boolNeuron.getMinTeachIndexes()
    boolNeuron.setTeachIndexes(minIndexes)
    boolNeuron.teach()
    print(minIndexes)
    print(boolNeuron.getLogStr())
    print()

    boolNeuron = BoolNeuron(function, False)
    minIndexes = boolNeuron.getMinTeachIndexes()
    boolNeuron.setTeachIndexes(minIndexes)
    boolNeuron.teach()
    print(minIndexes)
    print(boolNeuron.getLogStr())
