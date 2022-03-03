from NeuralNetworkPlot import NeuralNetworkPlot

if __name__ == '__main__':
    '''''
    y = exp(-0.1t^2)
    a = -5
    b = 5
    '''''

    leftWall = -5
    rightWall = 5
    pointsForRPM = 10

    neuralNetwork = NeuralNetworkPlot(leftWall, rightWall, 8000)
    neuralNetwork.plotFunction()
    neuralNetwork.teach()
    neuralNetwork.plotForecast()

    # Зафиксируем параметры кроме количества эпох
    for i in range(pointsForRPM):
        neuralNetwork = NeuralNetworkPlot(leftWall, rightWall, (i + 1) * 1500)
        print('M = ' + str(neuralNetwork.getMaxGenerations()) + ' | ' + str(neuralNetwork.teach()))
    print()

    # Зафиксируем параметры кроме размера окна
    for i in range(2, pointsForRPM + 2):
        neuralNetwork = NeuralNetworkPlot(leftWall, rightWall, 8000, 80, i)
        print('P = ' + str(neuralNetwork.getNeuronsAmount()) + ' | ' + str(neuralNetwork.teach()))
    print()

    # Зафиксируем параметры кроме нормы обучения
    for i in range(pointsForRPM):
        neuralNetwork = NeuralNetworkPlot(leftWall, rightWall, 8000, 80, 4, 0.01 * (pow(i + 1, 2)))
        print('N = ' + str(neuralNetwork.getNorm()) + ' | ' + str(neuralNetwork.teach()))
