from copy import deepcopy


# возвращает матрицу кодировки для данного числа
def getNumberCode(number):
    if number < 0 or number > 9:
        return None
    numberCode = {1: [[-1, 1, -1], [1, 1, -1], [-1, 1, -1], [-1, 1, -1], [1, 1, 1]],
                  2: [[1, 1, 1], [-1, -1, 1], [1, 1, 1], [-1, -1, 1], [1, 1, 1]],
                  3: [[1, 1, 1], [-1, -1, 1], [1, 1, 1], [-1, -1, 1], [1, 1, 1]],
                  4: [[1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, -1, 1], [-1, -1, 1]],
                  5: [[1, 1, 1], [1, -1, -1], [1, 1, 1], [-1, -1, 1], [1, 1, 1]],
                  6: [[1, 1, 1], [1, -1, -1], [1, 1, 1], [1, -1, 1], [1, 1, 1]],
                  7: [[1, 1, 1], [-1, -1, 1], [-1, -1, 1], [-1, -1, 1], [-1, -1, 1]],
                  8: [[1, 1, 1], [1, -1, 1], [1, 1, 1], [1, -1, 1], [1, 1, 1]],
                  9: [[1, 1, 1], [1, -1, 1], [1, 1, 1], [-1, -1, 1], [1, 1, 1]],
                  0: [[1, 1, 1], [1, -1, 1], [1, -1, 1], [1, -1, 1], [1, 1, 1]]}
    return deepcopy(numberCode.get(number))


# возвращает строковое представление матрицы закодированного объекта
def getStrMatrix(matrix):
    result = ""
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j] == 1:
                result += "*"
            else:
                result += " "
        result += '\n'
    return result


# возвращает вектор из матрицы
def getVectorFromMatrix(matrix):
    result = []
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            result.append(matrix[i][j])
    return result


class HopfildNetwork:
    def __init__(self, images):
        # количество обучающих образцов
        self.__imageNumber = len(images)
        # обучающие образцы
        self.__images = images
        # матричные кодировки обучающих образцов
        self.__matrixArray = self.__getMatrixArray()
        # матрица весов
        self.__weights = self.__getInitWeights()
        # количество пикселей в матрице каждого числа
        self.__amountPixels = self.__getAmountPixels()
        # массив векторных представлений кодировки обучающих образцов
        self.__vectorPatterns = self.__getVectorPatterns()

    # возвращает векторное представление кодировки обучающих векторов
    def __getVectorPatterns(self):
        result = []
        for i in range(self.__imageNumber):
            result.append(getVectorFromMatrix(self.__matrixArray[i]))
        return result

    # возвращает количество пикселей (размер матрицы запоминаемого объекта)
    def __getAmountPixels(self):
        return len(self.__matrixArray) * len(self.__matrixArray[0])

    # обнуляется главная диагональ весов
    def __zeroMainDiag(self):
        for i in range(len(self.__weights)):
            for j in range(len(self.__weights)):
                if i == j:
                    self.__weights[i][j] = 0

    # создается пустая матрица весов нужного размерв
    def __getInitWeights(self):
        result = []
        matrix_size = self.__imageNumber * len(self.__matrixArray[0])
        # заполняем матрицу пустотой
        for i in range(matrix_size):
            result.append([])
            for j in range(matrix_size):
                result[i].append([])
        return result

    # возвращает массив матриц закодированных объектов
    def __getMatrixArray(self):
        result = []
        for number in self.__images:
            result.append(getNumberCode(number))
        return result

    # возвращает отклик по номеру эпохи и номеру пикселя
    def __getOutput(self, ageNumber, pixelNumber):
        return self.__getActivationFunction(ageNumber, pixelNumber)

    # возвращает значение узла по номеру эпохи и номеру пикселя
    def __getNet(self, ageNumber, pixelNumber):
        result = 0
        for i in range(pixelNumber - 1):
            result += self.__weights[i][pixelNumber] * self.__getOutput(ageNumber, i)
        for i in range(pixelNumber, self.__amountPixels):
            result += self.__weights[i][pixelNumber] * self.__getOutput(ageNumber - 1, i)
        return result

    # возвращает значение функции акттивации по номеру эпохи и номеру пикселя
    def __getActivationFunction(self, ageNumber, pixelNumber):
        net = self.__getNet(ageNumber, pixelNumber)
        if net > 0:
            return 1
        if net < 0:
            return -1
        return self.__getActivationFunction(ageNumber, pixelNumber)

    # возвращает массив матриц закодированных объектов в строковом представлении
    def getMatrixArrayStr(self):
        result = ""
        for matrix in self.__matrixArray:
            result += getStrMatrix(matrix)
            result += "\n"
        return result

    # строковое представление класса - строковое представление матрицы весов и векторов обучающих образцов
    def __str__(self):
        result = ""
        for i in range(len(self.__weights)):
            for j in range(len(self.__weights[i])):
                result += str(self.__weights[i][j])
            result += '\n'
        result += '\n'
        for vector in self.__vectorPatterns:
            result += str(vector)
            result += '\n'
        return result
