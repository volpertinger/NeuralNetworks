from NeuralNetworkPlot import NeuralNetworkPlot

if __name__ == '__main__':
    '''''
    y = exp(-0.1t^2)
    a = -5
    b = 5
    '''''

    leftWall = -5
    rightWall = 5

    # Зафиксируем параметры кроме количества эпох

    neuralNetwork = NeuralNetworkPlot(leftWall, rightWall, 2000)
    neuralNetwork.plotFunction()
    print('M=2000: ' + str(neuralNetwork.teach()))
    neuralNetwork.plotForecast()

    neuralNetwork = NeuralNetworkPlot(leftWall, rightWall, 4000)
    print('M=4000: ' + str(neuralNetwork.teach()))
    neuralNetwork.plotForecast()

    neuralNetwork = NeuralNetworkPlot(leftWall, rightWall, 8000)
    print('M=8000: ' + str(neuralNetwork.teach()))
    neuralNetwork.plotForecast()

    neuralNetwork = NeuralNetworkPlot(leftWall, rightWall, 16000)
    print('M=16000: ' + str(neuralNetwork.teach()))
    neuralNetwork.plotForecast()

    # Зафиксируем параметры кроме нормы обучения

    neuralNetwork = NeuralNetworkPlot(leftWall, rightWall, 8000, 80, 4, 0.5)
    print('Norm=0.5: ' + str(neuralNetwork.teach()))
    neuralNetwork.plotForecast()

    neuralNetwork = NeuralNetworkPlot(leftWall, rightWall, 8000, 80, 4, 0.3)
    print('Norm=0.3: ' + str(neuralNetwork.teach()))
    neuralNetwork.plotForecast()

    neuralNetwork = NeuralNetworkPlot(leftWall, rightWall, 8000, 80, 4, 0.1)
    print('Norm=0.1: ' + str(neuralNetwork.teach()))
    neuralNetwork.plotForecast()

    neuralNetwork = NeuralNetworkPlot(leftWall, rightWall, 8000, 80, 4, 0.01)
    print('Norm=0.01: ' + str(neuralNetwork.teach()))
    neuralNetwork.plotForecast()
