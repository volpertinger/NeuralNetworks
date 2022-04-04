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
        self.__centerNumber, self.amountRBF = self.__getInitRBF()
        # индексы центров
        self.__centersRBF = self.__getCentersRBF()
        # веса и синоптические веса
        self.__weights, self.__synopticWeights = self.__getInitWeights()

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
        for i in range(self.__function):
            if self.__function[i] == self.__centerNumber:
                result.append(self.__function[i])
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
        for i in range(self.__centerNumber):
            result.append([])
            for j in range(self.__size):
                result[len(result) - 1].append(0)
        return result

    # возвращает синоптические веса (между центрами и функцией активации)
    def __getSynopticWeights(self):
        result = [0]
        for i in range(len(self.__function)):
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
    def __getSimpleActivationFunction(self):
        pass

    # возвращает значение сложной функции активации
    def __getComplexActivationFunction(self):
        pass

    # возвращает функцию активации (простую или сложную)
    def __getActivationFunction(self):
        return self.__getSimpleActivationFunction() if self.__isSimpleActivationFunction \
            else self.__getComplexActivationFunction()

    # возвращает значение в узле функции активации
    def __getNet(self, vector):
        result = self.__synopticWeights[0]
        for i in range(self.__centerNumber):
            result += self.__synopticWeights[i] * self.__getGaussPart(vector, i)
        return result

    # возвращает значение булевой функции в зависимости от значения узла функции активации
    def __getOutput(self, vector):
        return 1 if self.__getNet(vector) >= 0 else 0

    # возвращает дельту между полученным значением и тем, которое должно быть
    @staticmethod
    def getDelta(real, expected):
        return expected - real
