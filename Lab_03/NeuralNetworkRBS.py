import math
from copy import deepcopy


class NeuralNetworkRBS:
    class __Log:
        def __init__(self, gen, weights, function, delta):
            self.__gen = gen
            self.__weights = weights
            self.__function = function
            self.__delta = delta

        def __str__(self):
            return '-----Gen ' + str(self.__gen) + '\n' + 'Weights: ' + str(
                self.__weights) + '\n' + 'Function: ' + str(
                self.__function) + '\n' + 'Delta: ' + str(self.__delta) + '\n'

    def __init__(self, function, isSimpleActivationFunction, norm=0.3, teachIndexes=None):
        # вектор значений функции
        self.__function = function
        # количество переменных в функции
        self.__size = self.__getVariablesCount()
        # простая или сложная функция активации
        self.__isSimpleActivationFunction = isSimpleActivationFunction
        # норма обучения
        self.__norm = norm
        # число центров (0 или 1) и количество центров
        self.__centerNumber, self.__amountRBF = self.__getInitRBF()
        # индексы центров
        self.__centersRBF = self.__getCentersRBF()
        # синоптические веса
        self.__synopticWeights = self.__getSynopticWeights()
        # все возможные двоичные наборы переменных
        self.__variableSet = self.__getVariableSets()
        # лог обучения
        self.__log = []
        # дельты поколений
        self.__generationsDelta = []
        # индексы, по которым проводится обучение
        self.__teachIndexes = self.__getTeachIndexes(teachIndexes)
        # показатель обученности сети
        self.__isTrained = False

    def __addLog(self, weights, function):
        self.__log.append(
            self.__Log(len(self.__generationsDelta), weights, function,
                       self.__generationsDelta[len(self.__generationsDelta) - 1]))

    # получаем все двоичные наборы нашей функции
    def __getVariableSets(self):
        result = []
        for i in range(int(pow(2, self.__size))):
            result.append(self.__getBoolVector(i))
        return result

    def __getTeachIndexes(self, teachIndexes):
        if teachIndexes is None:
            result = []
            for i in range(len(self.__variableSet)):
                result.append(i)
            return result
        return teachIndexes

    # по номеру возвращает двоичный вектор
    def __getBoolVector(self, number):
        result = [0] * self.__size
        if number == 0:
            return result
        i = 0
        while number > 0:
            i += 1
            result[self.__size - i] = number % 2
            number = number // 2
        return result

    # возвращает число переменных функции
    def __getVariablesCount(self):
        return math.ceil(math.log2(len(self.__function)))

    # возвращает индексы центра
    def __getCentersRBF(self):
        result = []
        for i in range(len(self.__function)):
            if self.__function[i] == self.__centerNumber:
                result.append(self.__getBoolVector(i))
        return result

    # возвращает "0", если "0" меньше "1" и "1" иначе; min (количество "0", "1" в функции)
    def __getInitRBF(self):
        oneAmount = 0
        for i in range(len(self.__function)):
            oneAmount += self.__function[i]
        zeroAmount = len(self.__function) - oneAmount
        return (1, oneAmount) if oneAmount < zeroAmount else (0, zeroAmount)

    # возвращает синоптические веса (между центрами и функцией активации)
    def __getSynopticWeights(self):
        result = [0]
        for i in range(self.__amountRBF):
            result.append(0)
        return result

    # возвращает f(x) (зависит от весов и входного вектора)
    def __getGaussPart(self, vector, centerIndex):
        result = 0
        for i in range(self.__size):
            result += (vector[i] - self.__centersRBF[centerIndex][i]) ** 2
        result = math.exp(-1 * result)
        return result

    # возвращает значение простой функцию активации
    @staticmethod
    def __getSimpleActivationFunction(value):
        if value >= 0:
            return 1
        return 0

    # возвращает выход из выходов нейронов RBF
    def __getOutput(self, index):
        # константа
        result = self.__synopticWeights[self.__amountRBF]
        for i in range(self.__amountRBF):
            result += self.__synopticWeights[i] * self.__getGaussPart(self.__variableSet[index], i)
        return self.__getSimpleActivationFunction(result)

    def __makeCorrection(self, vector, delta):
        deltaNorm = self.__norm * delta
        # константа
        self.__synopticWeights[self.__amountRBF] += deltaNorm
        for i in range(self.__amountRBF):
            self.__synopticWeights[i] += deltaNorm * self.__getGaussPart(vector, i)
        return

    # возвращает веса для добавления в лог
    def __getWeightsForLog(self):
        result = [self.__synopticWeights[self.__amountRBF]]
        for i in range(self.__amountRBF):
            result.append(self.__synopticWeights[i])
        return result

    # возвращает выходную функцию для добавления в лог
    def __getFunctionForLog(self):
        result = deepcopy(self.__function)
        for i in range(len(self.__function)):
            delta = self.__getDelta(i)
            if delta != 0:
                result[i] -= delta
        return result

    # выполняет обучение по поколению (все наборы переменных)
    def __solveGeneration(self):
        generationDelta = 0
        currentWeights = self.__getWeightsForLog()
        currentFunction = self.__getFunctionForLog()
        for i in self.__teachIndexes:
            delta = self.__getDelta(i)
            if delta != 0:
                self.__makeCorrection(self.__variableSet[i], delta)
                generationDelta += 1
        self.__generationsDelta.append(generationDelta)
        self.__addLog(currentWeights, currentFunction)
        return generationDelta

    def __testAfterTeach(self):
        for i in range(len(self.__variableSet)):
            delta = self.__getDelta(i)
            if delta != 0:
                self.__isTrained = False
                return
        self.__isTrained = True

    # обучение нейронной сети
    def teach(self):
        generationDelta = self.__solveGeneration()
        while generationDelta > 0:
            generationDelta = self.__solveGeneration()
        self.__testAfterTeach()
        return True

    # возвращает вектор синоптических весов
    def getSynopticWeights(self):
        return self.__synopticWeights

    # возвращает лог в строковом представлении
    def getLogStr(self):
        result = ''
        for element in self.__log:
            result += str(element)
        result += '-----' + 'Trained: ' + str(self.__isTrained)
        return result

    # возвращает дельту
    def __getDelta(self, index):
        return self.__function[index] - self.__getOutput(index)
