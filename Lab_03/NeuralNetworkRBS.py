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

    def __init__(self, function, isSimpleFunction=True, norm=0.3, teachIndexes=None):
        # вектор значений функции
        self.__function = function
        # количество переменных в функции
        self.__size = self.__getVariablesCount()
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
        # индексы, по которым проводится проверка
        self.__testIndexes = self.__getTestIndexes()
        # показатель обученности сети
        self.__isTrained = False
        # простая или нет функция активации
        self.__isSimpleFunction = isSimpleFunction

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

    # возвращает индексы, на которых обучаем сеть
    def __getTeachIndexes(self, teachIndexes):
        if teachIndexes is None:
            result = []
            for i in range(len(self.__variableSet)):
                result.append(i)
            return result
        return teachIndexes

    # возвращает индексы, на которых сеть проверяется (и не обучается)
    def __getTestIndexes(self):
        result = []
        for i in range(len(self.__variableSet)):
            if self.__teachIndexes.count(i) == 0:
                result.append(i)
        return result

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
        if value > 0:
            return 1
        return 0

    # возвращает значение сложной функцию активации
    @staticmethod
    def __getComplexActivationFunction(value):
        return round(1 / 2 * (1 + (math.exp(value) - math.exp(-value)) / (math.exp(value) + math.exp(-value))))

    def __getActivationFunction(self, value):
        if self.__isSimpleFunction:
            return self.__getSimpleActivationFunction(value)
        return self.__getComplexActivationFunction(value)

    # возвращает выход из выходов нейронов RBF
    def __getOutput(self, index):
        # константа
        result = self.__synopticWeights[self.__amountRBF]
        for i in range(self.__amountRBF):
            result += self.__synopticWeights[i] * self.__getGaussPart(self.__variableSet[index], i)
        return self.__getSimpleActivationFunction(result)

    # коррекция синоптических весов
    def __makeCorrection(self, vector, delta, output):
        deltaNorm = self.__norm * delta
        # если функция активации сложная, умножим на производную
        if not self.__isSimpleFunction:
            deltaNorm *= 2 / (pow(math.exp(output) + math.exp(-output), 2))
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
                output = self.__getOutput(i)
                self.__makeCorrection(self.__variableSet[i], delta, output)
                generationDelta += 1
        self.__generationsDelta.append(generationDelta)
        self.__addLog(currentWeights, currentFunction)
        return generationDelta

    # проверяет функцию после обучения
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

    # возвращает сеть к исходному состоянию
    def __reset(self, savedTeachIndexes):
        self.__teachIndexes = deepcopy(savedTeachIndexes)
        self.__testIndexes = deepcopy(self.__getTestIndexes())
        self.__synopticWeights = [0.] * (self.__amountRBF + 1)
        self.__constantWeight = 0.
        self.__log = []
        self.__generationsDelta = []
        self.__isTrained = False

    # возвращает показатель обученности сети
    def isTrained(self):
        return self.__isTrained

    # получаем минимальный набор, на котором можно обучить сеть
    def getMinTeachIndexes(self):
        savedTeachIndexes = deepcopy(self.__teachIndexes)
        self.__testIndexes = self.__getTestIndexes()

        result = self.__teachIndexes
        currentSize = len(result)
        isTrained = True
        while currentSize > 0 and isTrained:
            isTrained = False
            for indexes in self.__getPossibleIndexes(currentSize):
                self.__reset(deepcopy(indexes))
                self.teach()
                if self.isTrained():
                    result = deepcopy(indexes)
                    currentSize -= 1
                    isTrained = True
                    break

        self.__reset(savedTeachIndexes)
        return result

    # устанавливаем индексы для обучения
    def setTeachIndexes(self, teachIndexes):
        if teachIndexes is None:
            return False

        for element in teachIndexes:
            if element < 0 or element >= len(self.__function):
                return False

        teachIndexes.sort()
        for i in range(len(teachIndexes) - 1):
            if teachIndexes[i] == teachIndexes[i + 1]:
                return False

        self.__teachIndexes = teachIndexes
        self.__testIndexes = self.__getTestIndexes()
        return True

    # возвращает всевозможные перестановки индексов
    def __getPossibleIndexes(self, size):
        result = [[]]
        for i in range(size):
            result[0].append(i)
        if size > len(self.__function):
            return result
        while True:
            lastIndexes = result[len(result) - 1]
            newIndexes = moveIndexes(lastIndexes, size - 1, len(self.__function))
            if newIndexes is not False:
                result.append(deepcopy(newIndexes))
                continue
            break
        return result


# перемещает индексы для нахождения перестановок
def moveIndexes(array, index, wall):
    if array[index] < wall - 1:
        array[index] += 1
        for i in range(index, len(array)):
            array[i] = array[index] + i - index
        return array
    if index == 0:
        return False
    return moveIndexes(array, index - 1, array[index])
