from math import log2, pow, floor, ceil
from copy import deepcopy


class BoolNeuron:
    class __Log:
        def __init__(self, gen, weights, constantWeight, function, delta):
            self.__gen = gen
            self.__weights = weights
            self.__constantWeight = constantWeight
            self.__function = function
            self.__delta = delta

        def __str__(self):
            return '-----Gen ' + str(self.__gen) + '\n' + 'Weights: ' + str(
                self.__weights) + '\n' + 'ConstantWeight: ' + str(self.__constantWeight) + '\n' + 'Function: ' + str(
                self.__function) + '\n' + 'Delta: ' + str(self.__delta) + '\n'

    def __init__(self, boolVector, isSimpleActivationFunction=True, teachFraction=1, norm=0.3):
        self.__isSimpleActivationFunction = isSimpleActivationFunction
        self.__norm = norm
        self.__constantWeight = 1
        self.__size = int(log2(len(boolVector)))
        self.__weights = [0.] * self.__size
        self.__boolVector = boolVector
        self.__variableSets = self.__getVariableSets()
        self.__generationsDelta = []
        self.__log = []
        self.__teachFraction = teachFraction
        self.__teachIndexes = self.__getTeachIndexes()
        self.__testIndexes = self.__getTestIndexes()
        self.__isTrained = False

    def __addLog(self, gen, weights, constantWeight, function, delta):
        self.__log.append(self.__Log(gen, weights, constantWeight, function, delta))

    def __getVariableSets(self):
        result = []
        for i in range(int(pow(2, self.__size))):
            result.append(self.__getBoolList(i))
        return result

    def __getBoolList(self, number):
        result = [0] * self.__size
        if number == 0:
            return result
        i = 0
        while number > 0:
            i += 1
            result[self.__size - i] = number % 2
            number = number // 2
        return result

    def __getTeachIndexes(self):
        result = []
        if not self.__isCorrectTeachFraction():
            for i in range(len(self.__variableSets)):
                result.append(i)
            return result
        numberOfTeachIndexes = ceil(len(self.__variableSets) * self.__teachFraction)
        for i in range(numberOfTeachIndexes):
            result.append(i)
        return result

    def __getTestIndexes(self):
        result = []
        for i in range(len(self.__variableSets)):
            if self.__teachIndexes.count(i) == 0:
                result.append(i)
        return result

    def __makeCorrection(self, delta, variableSet, net):
        deltaNorm = self.__norm * delta
        if not self.__isSimpleActivationFunction:
            deltaNorm *= 1 / (2 * pow((1 + abs(net)), 2))
        for i in range(self.__size):
            self.__weights[i] = self.__weights[i] + deltaNorm * variableSet[i]
            self.__constantWeight = self.__constantWeight + deltaNorm

    def __getNet(self, indexSet):
        result = self.__constantWeight
        for i in range(self.__size):
            result += self.__weights[i] * self.__variableSets[indexSet][i]
        return result

    def __getSimpleActivationFunction(self, indexSet):
        if self.__getNet(indexSet) > 0:
            return 1
        return 0

    def __getNotSimpleActivationFunction(self, indexSet):
        net = self.__getNet(indexSet)
        return round(0.5 * (net / (1 + abs(net)) + 1))

    def __getActivationFunction(self, indexSet):
        if self.__isSimpleActivationFunction:
            return self.__getSimpleActivationFunction(indexSet)
        return self.__getNotSimpleActivationFunction(indexSet)

    def __getDelta(self, indexSet):
        return self.__boolVector[indexSet] - self.__getActivationFunction(indexSet)

    def __solveGeneration(self):
        generationDelta = 0
        currentFunction = deepcopy(self.__boolVector)  # for log
        for i in self.__teachIndexes:
            delta = self.__getDelta(i)
            if delta != 0:
                currentFunction[i] = self.__boolVector[i] - delta  # for log
                generationDelta += 1
                self.__makeCorrection(delta, self.__variableSets[i], self.__getNet(i))
            else:
                currentFunction[i] = self.__boolVector[i]  # for log
        self.__generationsDelta.append(generationDelta)
        self.__addLog(len(self.__generationsDelta), self.__weights, self.__constantWeight, currentFunction,
                      generationDelta)
        return generationDelta

    def __isCorrectBoolVector(self):
        for element in self.__boolVector:
            if element != 0 and element != 1:
                return False
            return True

    def __isCorrectTeachFraction(self):
        if self.__teachFraction <= 0 or self.__teachFraction > 1:
            return False
        if floor(self.__teachFraction * len(self.__variableSets)) < 1:
            return False
        return True

    def __isCorrectData(self):
        if (0 < self.__norm <= 1) and (
                log2(
                    len(self.__boolVector)).is_integer()) and self.__isCorrectTeachFraction() and self.__isCorrectBoolVector():
            return True
        return False

    def __testAfterTeach(self):
        for i in self.__testIndexes:
            delta = self.__getDelta(i)
            if delta != 0:
                self.__isTrained = False
                return
        self.__isTrained = True

    def teach(self):
        if self.__isCorrectData():
            generationDelta = self.__solveGeneration()
            while generationDelta > 0:
                generationDelta = self.__solveGeneration()
            self.__testAfterTeach()
            return True
        return False

    def getLogStr(self):
        result = ''
        for element in self.__log:
            result += str(element)
        result += 'Trained: ' + str(self.__isTrained)
        return result

    def isTrained(self):
        return self.__isTrained

    def __str__(self):
        result = 'weights: ' + str(self.__weights) + '\n' + 'constant weight: ' + str(
            self.__constantWeight) + '\n' + 'generations delta: ' + str(
            self.__generationsDelta) + '\n'
        return result
