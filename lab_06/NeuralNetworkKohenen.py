from copy import deepcopy
from math import sqrt, pow


# возвращает из json массив с координатами
def getCoordinates(json_data):
    result = []
    for element in json_data:
        geodata = element.get("geodata_center")
        if geodata:
            coordinates = geodata.get("coordinates")
            if coordinates:
                result.append(deepcopy(coordinates))
    return result


# возвращает расстояние между двумя точками с двумя координатами
def getRange(lhs, rhs):
    result = sqrt(pow(lhs[0] - rhs[0], 2) + pow(lhs[1] - rhs[1], 2))
    return result


# возвращает словарь, key - административный округ Москвы, value - координаты
def getCoordinatesDictionary():
    dictionary = {"ЦАО": [37.621184, 55.753600],
                  "САО": [37.525774, 55.838390],
                  "СВАО": [37.632565, 55.854875],
                  "ВАО": [37.775631, 55.787715],
                  "ЮВАО": [37.754592, 55.692019],
                  "ЮАО": [37.678065, 55.622014],
                  "ЮЗАО": [37.576187, 55.662735],
                  "ЗАО": [37.443533, 55.728003],
                  "СЗАО": [37.451555, 55.829370],
                  "ЗеАО": [37.194250, 55.987583],
                  "ТрАО": [37.146999, 55.355802],
                  "НоАО": [37.370724, 55.558127]}
    return deepcopy(dictionary)


class NeuralNetwork:
    def __init__(self):
        # массив кластеров
        self.__centers = getCoordinatesDictionary()
        self.__clusters = self.__getInitClusters()

    # получаем ключ, ближе к которому находится вектор
    def __getClusterKey(self, coordinates):
        min_key = next(iter(self.__centers))
        min_value = getRange(coordinates, self.__centers[min_key])
        for key in self.__centers.keys():
            value = getRange(coordinates, self.__centers[key])
            if value < min_value:
                min_key = key
                min_value = value
        return min_key

    # получаем словарь кластеров, куда будем добавлять координаты входов
    def __getInitClusters(self):
        result = {}
        for key in self.__centers.keys():
            result.update({key: []})
        return result

    # вставляем координату в кластер
    def insert(self, coordinates):
        key = self.__getClusterKey(coordinates)
        self.__clusters.get(key).append(coordinates)

    # возвращаем словарь кластеров
    def getClusters(self):
        return self.__clusters
