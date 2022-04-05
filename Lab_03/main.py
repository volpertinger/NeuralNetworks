from NeuralNetworkRBS import NeuralNetworkRBS

if __name__ == '__main__':
    # (x1 + x2)x3 + x4
    # function = [0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, ]

    # (!(x1x2)x3x4
    function = [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]
    neuralNetwork = NeuralNetworkRBS(function, True)
    neuralNetwork.teach()
    print(neuralNetwork.getSynopticWeights())
