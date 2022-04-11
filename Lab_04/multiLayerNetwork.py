from math import exp


# значение функции активации
def getActivationFunction(net):
    return (1 - exp(-net)) / (1 + exp(-net))


# производная функции активации (для коррекции)
def getActivationDerivative(net):
    return (2 * exp(net)) / ((exp(net) + 1) ** 2)


class NeuralNetwork:
    def __init__(self, inputVector, outputVector, architecture, norm=0.3, maxError=0.1):
        # вектор входных сигналов
        self.__inputVector = inputVector
        # вектор выходной функции
        self.__outputVector = outputVector
        # количество переменных на входе
        self.__variableCount = architecture[0]
        # количество скрытых нейронов
        self.__hideNeuronCount = architecture[1]
        # количество открытых нейронов
        self.__openNeuronCount = architecture[2]
        # норма обучения
        self.__norm = norm
        # максимальное значение суммарной среднеквадратичной ошибки в эпохе
        self.__maxError = maxError
        # веса между переменными и скрытым слоем
        self.__hideWeights = self.__getHideWeights()
        # веса между скрытым слоем и открытым слоем
        self.__openWeights = self.__getOpenWeights()

    # веса между входами и нейронами скрытого слоя
    def __getHideWeights(self):
        result = []
        for i in range(self.__hideNeuronCount):
            # в индексе 0 - вес константы
            result.append([0])
            for j in range(self.__variableCount):
                result[len(result) - 1].append(0)
        return result

    # веса между скрытым и открытым слоями
    def __getOpenWeights(self):
        result = []
        for i in range(self.__openNeuronCount):
            # в индексе 0 - вес константы
            result.append([0])
            for j in range(self.__hideNeuronCount):
                result[len(result) - 1].append(0)
        return result

    # считаем значение выхода в скрытый нейрон заданного индекса
    def __getHideOutput(self, index):
        result = 0
        # + 1 из-за учета константы (костыль, но зато хорошо и удобно сравнивать с методичкой)
        for i in range(self.__variableCount + 1):
            result += self.__hideWeights[index][i] * self.__inputVector[i]
        return getActivationFunction(result)

    # считаем значение выхода в открытый нейрон заданного индекса
    def __getOpenOutput(self, index):
        result = 0
        # + 1 т.к. здесь тоже учитываются константы
        for i in range(self.__hideNeuronCount + 1):
            result += self.__openWeights[index][i] * self.__getHideOutput(i)
        return getActivationFunction(result)

    # получаем дельту по индексу открытого нейрона для коррекции весов между скрытым и открытым слоями
    def __getDeltaOpen(self, index):
        return self.__outputVector[index] - self.__getOpenOutput(index)

    # получаем дельту по индексу скрытого нейрона для коррекции веса между входными переменными и скрытым слоем
    # deltaArray - массив дельт между выходным вектором и открытыми нейронами
    # oldWeightArray - массив массивов с весами между скрытым и открытым слоем (до коррекции)
    def __getDeltaHide(self, index, deltaArray, oldWeightArray):
        result = 0
        for i in range(self.__openNeuronCount):
            result += deltaArray[i] * oldWeightArray[index][i]
        return result

    # выполняется коррекция весов между скрытым и открытым слоем по дельте и индексу
    def __makeOpenCorrection(self, index, delta):
        deltaNorm = delta * self.__norm
        # коррекция константы
        self.__openWeights[index][0] += deltaNorm
        for i in range(self.__hideNeuronCount):
            # + 1 - смещение от константы
            self.__openWeights[index][i + 1] += deltaNorm * getActivationDerivative(
                self.__getOpenOutput(index)) * self.__getHideOutput(i)

    # выполняется коррекция весов между входами и скрытым слоем
    def __makeHideCorrection(self, index, deltaArray, oldWeightArray):
        deltaNorm = self.__getDeltaHide(index, deltaArray, oldWeightArray) * self.__norm
        # коррекция константы
        self.__hideWeights[index][0] += deltaNorm
        for i in range(self.__variableCount):
            # + 1 - смещение от константы
            self.__hideWeights[index][i + 1] += deltaNorm * getActivationDerivative(self.__getOpenOutput(index)) * \
                                                self.__inputVector[i + 1]
