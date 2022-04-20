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

    def __getAmountPixels(self):
        return len(self.__matrixArray) * len(self.__matrixArray[0])

    def __getInitWeights(self):
        result = []
        for i in range(self.__imageNumber):
            for j in range(len(self.__matrixArray[i])):
                result.append([])
                for k in range(len(self.__matrixArray[i][j])):
                    result[(i + 1) * j].append(self.__matrixArray[i][j])
        return result

    def __getMatrixArray(self):
        result = []
        for number in self.__images:
            result.append(getNumberCode(number))
        return result

    def __getOutput(self, ageNumber, pixelNumber):
        return self.__getActivationFunction(ageNumber, pixelNumber)

    def __getNet(self, ageNumber, pixelNumber):
        result = 0
        for i in range(pixelNumber - 1):
            result += self.__weights[i][pixelNumber] * self.__getOutput(ageNumber, i)
        for i in range(pixelNumber, self.__amountPixels):
            result += self.__weights[i][pixelNumber] * self.__getOutput(ageNumber - 1, i)
        return result

    def __getActivationFunction(self, ageNumber, pixelNumber):
        net = self.__getNet(ageNumber, pixelNumber)
        if net > 0:
            return 1
        if net < 0:
            return -1
        return self.__getActivationFunction(ageNumber, pixelNumber)

    def getMatrixArrayStr(self):
        result = ""
        for matrix in self.__matrixArray:
            result += getStrMatrix(matrix)
            result += "\n"
        return result

    def __str__(self):
        result = ""
        for i in range(len(self.__weights)):
            for j in range(len(self.__weights[i])):
                result += str(self.__weights[i][j])
        return result
