from math import exp, pow, sqrt

import matplotlib.pyplot as plt

from copy import deepcopy


def standardFunction(x):
    return exp(-0.1 * pow(x, 2))


class NeuralNetworkPlot:
    class __Dot:
        def __init__(self, x, y):
            self.x = x
            self.y = y

        def __str__(self):
            return '(' + str(self.x) + '|' + str(self.y) + ')'

    def __init__(self, leftWall, rightWall, maxGenerations=None, pointsAmount=80, neuronsAmount=4, norm=0.01,
                 maxDelta=0.05, isGoToZero=True, function=standardFunction):
        self.__leftWall = leftWall
        self.__rightWall = rightWall
        self.__pointsAmount = pointsAmount
        self.__neuronsAmount = neuronsAmount
        self.__function = function
        self.__dots = self.__getDots()
        self.__forecastDots = []
        self.__maxDelta = maxDelta
        self.__maxGenerations = maxGenerations
        self.__norm = norm
        self.__weights = [0] * (neuronsAmount + 1)
        self.__isGoToZero = isGoToZero

    def __str__(self):
        return str(self.__weights)

    def __getDots(self):
        result = []
        delta = (self.__rightWall - self.__leftWall) / self.__pointsAmount
        for i in range(self.__pointsAmount):
            x = self.__leftWall + delta * i
            result.append(self.__Dot(x, self.__function(x)))
        return result

    def __setForecastDots(self):
        delta = (self.__rightWall - self.__leftWall) / self.__pointsAmount
        for i in range(self.__pointsAmount):
            x = self.__dots[len(self.__dots) - 1].x + delta * i
            self.__forecastDots.append(self.__Dot(x, self.__getNet(i + self.__pointsAmount)))

    def __isRightArguments(self):
        if (self.__rightWall < self.__leftWall) or (self.__maxDelta < 0):
            return False
        return True

    def __solveGeneration(self):
        rpm = 0  # квадрат среднеквадратичной погрешности
        rpmMax = 0
        for i in range(self.__neuronsAmount, self.__pointsAmount):
            forecastDot = self.__getNet(i)
            delta = self.__dots[i].y - forecastDot
            rpm += pow(delta, 2)
            if sqrt(rpm) > self.__maxDelta:
                rpmMax = sqrt(rpm)
                rpm = 0
                self.__makeCorrection(delta, i)
        return rpmMax

    def __teachByMaxGenerations(self):
        for i in range(self.__maxGenerations):
            self.__solveGeneration()

    def __teachByMaxDelta(self):
        generationDelta = self.__solveGeneration()
        while generationDelta > self.__maxDelta:
            generationDelta = self.__solveGeneration()

    def plotForecast(self):
        yDots = []
        xDots = []
        for dot in self.__dots:
            yDots.append(dot.y)
            xDots.append(dot.x)
        plt.plot(xDots, yDots, label='function')
        yDots = []
        xDots = []
        for dot in self.__forecastDots:
            yDots.append(dot.y)
            xDots.append(dot.x)
        plt.plot(xDots, yDots, label='forecast')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plotFunction(self):
        yDots = []
        xDots = []
        dots = deepcopy(self.__dots)
        if len(dots) < 2:
            return False
        delta = dots[1].x - dots[0].x
        for i in range(len(dots)):
            x = self.__rightWall + (i + 1) * delta
            dots.append(self.__Dot(x, self.__function(x)))
        for dot in dots:
            yDots.append(dot.y)
            xDots.append(dot.x)
        plt.plot(xDots, yDots, label='function')
        plt.legend()
        plt.grid(True)
        plt.show()

    def teach(self):
        if not self.__isRightArguments():
            return False
        if self.__maxGenerations is not None:
            self.__teachByMaxGenerations()
        else:
            self.__teachByMaxDelta()
        self.__setForecastDots()
        return True

    def __getNet(self, index):
        result = 0
        if not self.__isGoToZero:
            result += self.__weights[0]
        for i in range(self.__neuronsAmount):
            if index + i - self.__neuronsAmount < len(self.__dots):
                result += self.__weights[i + 1] * self.__dots[index + i - self.__neuronsAmount].y
                continue
            if index + i - self.__neuronsAmount < len(self.__dots) + len(self.__forecastDots):
                result += self.__weights[i + 1] * self.__forecastDots[
                    index + i - self.__neuronsAmount - len(self.__dots)].y
        return result

    def __makeCorrection(self, delta, index):
        deltaNorm = self.__norm * delta
        self.__weights[0] += deltaNorm
        for i in range(self.__neuronsAmount):
            self.__weights[i + 1] += deltaNorm * self.__dots[index + i - self.__neuronsAmount].y
        return
