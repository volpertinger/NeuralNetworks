from BoolNeuron import BoolNeuron

if __name__ == '__main__':
    '''''
     __   __   __  __   __
    (X1 + X2 + X3)(X2 + X3 + X4)
    
    F = 1 1 1 1 1 1 0 1 1 1 1 1 1 1 0 0 
    '''''

    boolNeuron = BoolNeuron([1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0], False)
    if boolNeuron.teach():
        print(boolNeuron)
        print(boolNeuron.getLog())
    else:
        print('Error')
