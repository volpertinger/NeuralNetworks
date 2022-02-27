from math import exp, pow, sqrt
from copy import deepcopy


def standardFunction(x):
    return exp(-0.1 * pow(x, 2))


class NeuralNetworkPlot:

    def __init__(self, leftWall, rightWall, pointsAmount=20, neuronsAmount=4, norm=1, function=standardFunction):
        self.__leftWall = leftWall
        self.__rightWall = rightWall
        self.__pointsAmount = pointsAmount
        self.__neuronsAmount = neuronsAmount
        self.__dots = self.__getDots()
        self.__function = function
        self.__norm = norm
        self.__weights = [0] * (neuronsAmount + 1)  # примерно тут я решил не говнокодить и в индексе 0 - вес константы

    def __getDots(self):
        result = []
        delta = (self.__rightWall - self.__leftWall + 1) / self.__pointsAmount
        for i in range(self.__pointsAmount):
            result.append(self.__leftWall + delta * i)
        return result

    def __isRightArguments(self):
        if self.__rightWall < self.__leftWall:
            return False
        return True

    def __getForecastDot(self, index):
        result = self.__weights[0]
        for i in range(self.__neuronsAmount):
            if index - self.__neuronsAmount + i < 0:
                continue
            result += self.__weights[i + 1] * self.__dots[index - self.__neuronsAmount + i]
        return result

    def __getDelta(self, forecastDots):
        result = deepcopy(self.__dots)
        for element in forecastDots:
            result -= element
        return result

    def __getNet(self, index):
        result = self.__weights[0]
        for i in range(self.__neuronsAmount):
            result += self.__weights[i + 1] * self.__dots[index + i]

    def __makeCorrection(self, delta, index):
        self.__weights[0] += self.__norm * delta
        for i in range(self.__neuronsAmount):
            self.__weights[i + 1] += self.__norm * delta * self.__dots[index + i]

    @staticmethod
    def __getRMSDelta(lhs, rhs):
        result = 0
        for i in range(len(lhs)):
            result += pow(lhs[i] + rhs[i], 2)
        result = sqrt(result)
        return result
