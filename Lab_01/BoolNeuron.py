from math import log2, pow


class BoolNeuron:
    def __init__(self, boolVector, isSimpleActivationFunction=True, norm=0.3):
        self.__isSimpleActivationFunction = isSimpleActivationFunction
        self.__norm = norm
        self.__constantWeight = 1
        self.__size = int(log2(len(boolVector)))
        self.__weights = [0.] * self.__size
        self.__boolVector = boolVector
        self.__variableSets = self.__getVariableSets()
        self.__generationsDelta = []
        self.__log = ''

    def __addLog(self, isFinal, *data):
        for element in data:
            self.__log += str(element)
            self.__log += '     '
        if isFinal:
            self.__log += '\n'

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
        currentFunction = []  # for log
        for i in range(int(pow(2, self.__size))):
            delta = self.__getDelta(i)
            if delta != 0:
                currentFunction.append(self.__boolVector[i] - delta)  # for log
                generationDelta += 1
                self.__makeCorrection(delta, self.__variableSets[i], self.__getNet(i))
            else:
                currentFunction.append(self.__boolVector[i])  # for log
        self.__generationsDelta.append(generationDelta)
        self.__addLog(True, 'Gen: ', len(self.__generationsDelta), 'Weights: ', self.__weights, 'ConstantWeight: ',
                      self.__constantWeight, 'Function: ',
                      currentFunction, 'Delta: ', generationDelta)
        return generationDelta

    def __isCorrectBoolVector(self):
        for element in self.__boolVector:
            if element != 0 and element != 1:
                return False
            return True

    def __isCorrectData(self):
        if (0 < self.__norm <= 1) and (log2(len(self.__boolVector)).is_integer()) and self.__isCorrectBoolVector():
            return True
        return False

    def teach(self):
        if self.__isCorrectData():
            generationDelta = self.__solveGeneration()
            while generationDelta > 0:
                generationDelta = self.__solveGeneration()
            return True
        return False

    def getLog(self):
        return self.__log

    def __str__(self):
        result = 'weights: ' + str(self.__weights) + '\n' + 'constant weight: ' + str(
            self.__constantWeight) + '\n' + 'generations delta: ' + str(
            self.__generationsDelta) + '\n'
        return result
