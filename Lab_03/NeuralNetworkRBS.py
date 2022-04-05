import math


class NeuralNetworkRBS:
    def __init__(self, function, isSimpleActivationFunction, norm=0.3):
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
        # веса и синоптические веса
        self.__weights, self.__synopticWeights = self.__getInitWeights()
        # все возможные двоичные наборы переменных
        self.__variableSet = self.__getVariableSets()

    # получаем все двоичные наборы нашей функции
    def __getVariableSets(self):
        result = []
        for i in range(int(pow(2, self.__size))):
            result.append(self.__getBoolVector(i))
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
                result.append(i)
        return result

    # возвращает "0", если "0" меньше "1" и "1" иначе; min (количество "0", "1" в функции)
    def __getInitRBF(self):
        oneAmount = 0
        for i in range(len(self.__function)):
            oneAmount += self.__function[i]
        zeroAmount = len(self.__function) - oneAmount
        return (1, oneAmount) if oneAmount < zeroAmount else (0, zeroAmount)

    # возвращает массив массивов весов (между переменными и центрами)
    def __getWeights(self):
        result = []
        for i in range(self.__amountRBF):
            result.append([])
            for j in range(self.__size + 1):
                result[len(result) - 1].append(0)
        return result

    # возвращает синоптические веса (между центрами и функцией активации)
    def __getSynopticWeights(self):
        result = [0]
        for i in range(self.__amountRBF):
            result.append(0)
        return result

    # возвращает веса между переменными и центрами; синоптические веса
    def __getInitWeights(self):
        return self.__getWeights(), self.__getSynopticWeights()

    # возвращает f(x) (зависит от весов и входного вектора)
    def __getGaussPart(self, vector, centerIndex):
        result = 0
        for i in range(self.__size):
            result += pow(vector[i] - self.__weights[centerIndex][i], 2)
        result = math.exp(-1 * result)
        return result

    # возвращает значение простой функцию активации
    @staticmethod
    def __getSimpleActivationFunction(value):
        if value >= 0:
            return 1
        return 0

    # возвращает значение сложной функции активации
    def __getComplexActivationFunction(self, vector):
        net = self.__getNet(vector)
        return round(1 / 2 * (1 + (math.exp(net) - math.exp(-net)) / (math.exp(net) + math.exp(-net))))

    # возвращает функцию активации (простую или сложную)
    def __getActivationFunction(self, vector):
        return self.__getSimpleActivationFunction(vector) if self.__isSimpleActivationFunction \
            else self.__getComplexActivationFunction(vector)

    # возвращает значение в узле функции активации
    def __getNet(self, vector):
        # константа - последняя
        result = self.__synopticWeights[self.__amountRBF]
        for i in range(self.__amountRBF):
            result += self.__synopticWeights[i] * self.__getGaussPart(vector, i)
        return result

    # возвращает значение булевой функции в зависимости от значения узла функции активации
    def __getOutput_old(self, index):
        return self.__getActivationFunction(self.__variableSet[index])

    # возвращает входное значение в RBS нейрон
    def __getRBSNeuronInput(self, centerIndex, vector):
        # константа
        result = self.__weights[centerIndex][self.__size]
        for i in range(self.__size):
            result += vector[i] * self.__weights[centerIndex][i]
        return result

    # возвращает выход с RBS нейрона
    def __getRBSNeuronOutput(self, centerIndex, vector):
        return self.__getGaussPart(vector,centerIndex)

    # возвращает вектор входов из RBS нейронов
    def __getRBSVector(self, vector):
        result = []
        for i in range(self.__amountRBF):
            result.append(self.__getRBSNeuronOutput(i, vector))
        return result

    # возвращает выход из выходов нейронов RBF
    def __getOutput(self, index):
        # константа
        result = self.__synopticWeights[self.__amountRBF]
        vectorRBF = self.__getRBSVector(self.__variableSet[index])
        for i in range(self.__amountRBF):
            result += vectorRBF[i] * self.__synopticWeights[i]
        return self.__getSimpleActivationFunction(result)

    # возвращает дельту синоптического веса по дельте, вектору и индексу центра
    def __getDeltaSynopticWeight(self, delta, vector, centerIndex):
        return self.__norm * delta * self.__getGaussPart(vector, centerIndex)

    # выполняет коррекцию синоптического веса по дельте, вектору, индексу центра
    def __makeSynopticCorrectionIndex(self, delta, vector, centerIndex):
        self.__synopticWeights[centerIndex] += self.__getDeltaSynopticWeight(delta, vector, centerIndex)

    # выполняет коррекцию всех синоптических весов по вектору и дельте
    def __makeSynopticCorrectionAll(self, vector, delta):
        for i in range(self.__amountRBF):
            self.__makeSynopticCorrectionIndex(delta, vector, i)

    def __makeSynopticCorrectionConstant(self, delta):
        # последний - вес константы
        self.__synopticWeights[self.__amountRBF] += self.__norm * delta

    # возвращает дельту для RBS веса от нейрона RBS до переменной
    def __getDeltaRBS(self, delta, vector, variableIndex):
        return self.__norm * delta * vector[variableIndex]

    # производит коррекцию одного веса от переменной до нейрона RBS
    def __makeCorrectionRBSIndex(self, delta, vector, centerIndex, variableIndex):
        self.__weights[centerIndex][variableIndex] += self.__getDeltaRBS(delta, vector, variableIndex)

    # производит коррекцию всех весов от переменной до нейрона RBS
    def __makeCorrectionRBSAll(self, delta, vector, centerIndex):
        for i in range(self.__size):
            self.__makeCorrectionRBSIndex(delta, vector, centerIndex, i)

    # выполняет коррекцию синоптических весов и весов от переменной до RBS нейронов
    def __makeCorrection_old(self, vector, delta):
        for i in range(self.__amountRBF):
            self.__makeSynopticCorrectionIndex(delta, vector, i)
            self.__makeCorrectionRBSAll(delta, vector, i)
        self.__makeSynopticCorrectionConstant(delta)

    def __makeCorrection(self, vector, delta):
        deltaNorm = self.__norm * delta
        # константа
        self.__synopticWeights[self.__amountRBF] += deltaNorm
        for i in range(self.__amountRBF):
            outputRBS = self.__getRBSNeuronOutput(i, vector)
            self.__synopticWeights[i] += deltaNorm * outputRBS
            deltaNormRBS = deltaNorm * self.__getGaussPart(vector, i)
            # константа
            self.__weights[i][self.__size] += deltaNormRBS
            # коррекция весов от переменных к нейронам RBF
            for j in range(self.__size):
                self.__weights[i][j] += deltaNormRBS * vector[j]
        return

    # выполняет обучение по поколению (все наборы переменных)
    def __solveGeneration(self):
        generationDelta = 0
        for i in range(len(self.__variableSet)):
            delta = self.__getDelta(i)
            if delta != 0:
                self.__makeCorrection(self.__variableSet[i], delta)
                generationDelta += 1
        return generationDelta

    # обучение нейронной сети
    def teach(self):
        generationDelta = self.__solveGeneration()
        while generationDelta != 0:
            generationDelta = self.__solveGeneration()
        return True

    # возвращает вектор синоптических весов
    def getSynopticWeights(self):
        return self.__synopticWeights

    # возвращает дельту
    def __getDelta(self, index):
        return self.__function[index] - self.__getOutput(index)
