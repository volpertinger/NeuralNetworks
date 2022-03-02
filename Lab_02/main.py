from NeuralNetworkPlot import NeuralNetworkPlot

if __name__ == '__main__':
    '''''
    y = exp(-0.1t^2)
    a = -5
    b = 5
    '''''

    leftWall = -5
    rightWall = 5

    neuralNetwork = NeuralNetworkPlot(leftWall, rightWall, 2000)
    neuralNetwork.plotFunction()
    neuralNetwork.teach()
    neuralNetwork.plotForecast()

    neuralNetwork = NeuralNetworkPlot(leftWall, rightWall, 4000)
    neuralNetwork.teach()
    neuralNetwork.plotForecast()

    neuralNetwork = NeuralNetworkPlot(leftWall, rightWall, 8000)
    neuralNetwork.teach()
    neuralNetwork.plotForecast()

    neuralNetwork = NeuralNetworkPlot(leftWall, rightWall, 16000)
    neuralNetwork.teach()
    neuralNetwork.plotForecast()
