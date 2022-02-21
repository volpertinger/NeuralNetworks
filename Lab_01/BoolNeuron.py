from math import log2, pow


class BoolNeuron:
    def __init__(self, boolVector):
        self.__weights = [0] * len(boolVector)
        self.__size = int(log2(len(boolVector)))
        self.__constantWeight = 1
        self.__variableSets = self.__getVariableSets()

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

    def __str__(self):
        result = 'weights: ' + str(self.__weights) + '\n' + 'size: ' + str(self.__size) + '\n' + 'variableSets: ' + str(
            self.__variableSets)
        return result
