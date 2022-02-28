from NeuralNetworkPlot import NeuralNetworkPlot

if __name__ == '__main__':
    '''''
    y = exp(-0.1t^2)
    a = -5
    b = 5
    '''''
    leftWall = -5
    rightWall = 5
    neuralNetwork = NeuralNetworkPlot(leftWall, rightWall,10000)
    #neuralNetwork.plotFunction()
    neuralNetwork.teach()
    neuralNetwork.plotForecast()
