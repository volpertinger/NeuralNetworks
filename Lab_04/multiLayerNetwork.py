from math import exp, sqrt
from copy import deepcopy


# значение функции активации
def getActivationFunction(net):
    return (1 - exp(-net)) / (1 + exp(-net))


# производная функции активации (для коррекции)
def getActivationDerivative(net):
    return (2 * exp(net)) / ((exp(net) + 1) ** 2)


class NeuralNetwork:
    class __Log:
        def __init__(self, gen, weightsOpen, weightsHide, function, delta):
            self.__gen = gen
            self.__weightsOpen = weightsOpen
            self.__weightsHide = weightsHide
            self.__function = function
            self.__delta = delta

        def __str__(self):
            return '-----Gen ' + str(self.__gen) + '\n' + 'WeightsOpen: ' + str(
                self.__weightsOpen) + '\n' + 'WeightsHide: ' + str(self.__weightsHide) + '\n' + 'Function: ' + str(
                self.__function) + '\n' + 'Delta: ' + str(self.__delta) + '\n'

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
        # показатель обученности сети
        self.__isTrained = False
        # лог обучение
        self.__log = []

    # добавляется лог эпохи
    def __addLog(self, weightsOpen, weightsHide, function):
        gen = len(self.__log)
        delta = self.__getError()
        self.__log.append(self.__Log(gen, weightsOpen, weightsHide, function, delta))

    # функция выхода для логирования
    def __getCurrentFunction(self):
        result = []
        for i in range(self.__openNeuronCount):
            result.append(self.__getOpenOutput(i))
        return result

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
        result = self.__openWeights[index][0]
        for i in range(self.__hideNeuronCount):
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

    # выполняется коррекция весов между всеми слоями
    def __makeCorrection(self):
        # заполняем массив дельт и сохраняем веса между скрытым и открытым слоями, чтобы потом произвести
        # коррекцию весов между переменными и скрытым слоем
        deltaArray = []
        for i in range(self.__openNeuronCount):
            deltaArray.append(self.__getDeltaOpen(i))
        oldWeightsArray = deepcopy(self.__openWeights)

        # коррекция весов между скрытым и открытым слоем
        for i in range(self.__openNeuronCount):
            self.__makeOpenCorrection(i, deltaArray[i])

        # коррекция весов между переменными и скрытым слоем
        for i in range(self.__hideNeuronCount):
            self.__makeHideCorrection(i, deltaArray, oldWeightsArray)

    # получение среднеквадратичной ошибки эпохи
    def __getError(self):
        totalError = 0
        for i in range(self.__openNeuronCount):
            totalError += self.__getDeltaOpen(i) ** 2
        totalError = sqrt(totalError)
        return totalError

    # Поколение обучения. True - обучение завершено, False - не завершено
    def __solveGeneration(self):
        totalError = self.__getError()
        self.__addLog(self.__openWeights, self.__hideWeights, self.__getCurrentFunction())
        if totalError > self.__maxError:
            self.__makeCorrection()
            return False
        return True

    # производит полное обучение сети
    def teach(self):
        self.__isTrained = self.__solveGeneration()
        while not self.__isTrained:
            self.__isTrained = self.__solveGeneration()
        return True

    def getLogStr(self):
        result = ''
        for element in self.__log:
            result += str(element)
        result += '-----' + 'Trained: ' + str(self.__isTrained)
        return result
