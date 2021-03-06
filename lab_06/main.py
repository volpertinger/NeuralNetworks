import NeuralNetworkKohenen
import json

# Алгоритм: НС Кохенена

# Исходные кластеризуемые данные:
# Выборка взрослых больниц г.Москвы. Координаты местоположения (X, Y) или количество коек для больных

# p: Принадлежность округу Москвы (евклидово расстояния до координат (X0, Y0) центра округа)
# или количество коек для больных

if __name__ == '__main__':
    data_path = "data-5609-2022-04-04.json"
    data = open(data_path, 'r', encoding='windows-1251')
    json_data = json.load(data)
    data.close()
    hospitals = NeuralNetworkKohenen.getHospitals(json_data)

    neuralNetwork = NeuralNetworkKohenen.NeuralNetwork()
    for hospital in hospitals:
        neuralNetwork.insert(hospital)

    clusters = neuralNetwork.getClusters()

    print("Initial hospitals:")
    for hospital in hospitals:
        print(hospital)
    print()

    print("Cluster`s centers coordinates")
    cluster_dictionary = NeuralNetworkKohenen.getCoordinatesDictionary()
    for key in cluster_dictionary:
        print(key, ":")
        print(cluster_dictionary.get(key))
    print()

    print("After clustering")
    for key in clusters:
        print(key, ": ")
        for element in clusters.get(key):
            print(element)
        print()
    print()

    print("Issues rate")
    print(neuralNetwork.countIssues())
