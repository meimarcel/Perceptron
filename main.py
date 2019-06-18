import Perceptron
import numpy
import csv

PATH = "iris.csv"

perceptron = Perceptron.Perceptron(path=PATH, number_features=2, col1=0, col2=1, learning_rate=0.01)

perceptron.start_train()

taxa = perceptron.showAccuracy()
print('Taxa de Acertos: %.2f%%' % (taxa*100))

perceptron.plot()


