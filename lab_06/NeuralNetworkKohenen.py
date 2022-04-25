from copy import deepcopy
from math import sqrt, pow


class Hospital:
    def __init__(self, name, coordinates, area):
        # название больницы
        self.name = name
        # административный округ больницы
        self.area = area
        # координаты больницы
        self.coordinates = coordinates

    def __str__(self):
        return str(self.name) + ", " + self.area + ": " + str(self.coordinates)


# возвращает из json массив с координатами
def getHospitals(json_data):
    result = []
    for element in json_data:
        coordinates = element.get("geodata_center")
        name = element.get("ShortName")
        area = element.get("ObjectAddress")
        if coordinates:
            coordinates = coordinates.get("coordinates")
        if area:
            if len(area):
                area = area[0]
                if area:
                    area = area.get("AdmArea")
        if coordinates and name and area:
            result.append(Hospital(name, coordinates, area))
    return result


# возвращает расстояние между двумя точками с двумя координатами
def getRange(lhs, rhs):
    result = sqrt(pow(lhs[0] - rhs[0], 2) + pow(lhs[1] - rhs[1], 2))
    return result


# возвращает словарь, key - административный округ Москвы, value - координаты
def getCoordinatesDictionary():
    dictionary = {"Центральный административный округ": [37.621184, 55.753600],
                  "Северный административный округ": [37.525774, 55.838390],
                  "Северо-Восточный административный округ": [37.632565, 55.854875],
                  "Восточный административный округ": [37.775631, 55.787715],
                  "Юго-Восточный административный округ": [37.754592, 55.692019],
                  "Южный административный округ": [37.678065, 55.622014],
                  "Юго-Западный административный округ": [37.576187, 55.662735],
                  "Западный административный округ": [37.443533, 55.728003],
                  "Северо-Западный административный округ": [37.451555, 55.829370],
                  "Зеленоградский административный округ": [37.194250, 55.987583],
                  "Троицкий административный округ": [37.146999, 55.355802],
                  "Новомосковский административный округ": [37.370724, 55.558127]}
    return deepcopy(dictionary)


class NeuralNetwork:
    def __init__(self):
        # массив кластеров
        self.__centers = getCoordinatesDictionary()
        self.__clusters = self.__getInitClusters()

    # получаем ключ, ближе к которому находится вектор
    def __getClusterKey(self, hospital):
        min_key = next(iter(self.__centers))
        min_value = getRange(hospital.coordinates, self.__centers[min_key])
        for key in self.__centers.keys():
            value = getRange(hospital.coordinates, self.__centers[key])
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
    def insert(self, hospital):
        key = self.__getClusterKey(hospital)
        self.__clusters.get(key).append(hospital)

    # возвращаем словарь кластеров
    def getClusters(self):
        return self.__clusters

    # считаем процент ошибок
    def countIssues(self):
        # количество ошибок
        issues = 0
        # всего элементов
        total = 0
        for cluster in self.__clusters.keys():
            for value in self.__clusters.get(cluster):
                if value.area != cluster:
                    issues += 1
                total += 1
        return issues / total
