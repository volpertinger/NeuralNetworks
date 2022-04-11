from multiLayerNetwork import NeuralNetwork

if __name__ == '__main__':
    # 2-1-2 input: (1,1,2) 10 * output: (2,2) -> output^ (0.2, 0.2)
    architecture = [2, 1, 2]
    inputVector = [1, 1, 2]
    outputVector = [0.2, 0.2]
    neuralNetwork = NeuralNetwork(inputVector, outputVector, architecture, 0.3, 0.001)
    neuralNetwork.teach()
    print(neuralNetwork.getLogStr())
