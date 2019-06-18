# PERCEPTRON
# Codigo feito por @Mei Marcel
# Esse codigo utiliza das bibliotecas (numpy), (matplotlib.pyplot) e (csv)
# As bibliotecas (numpy) e (matplotlib.pyplot) talvez precisem ser baixadas
# O codigo pode funcionar sem a biblioteca (matplotlib.pyplot), porem a funcao "plot" nao podera ser usada, para isso, comente a linha da biblioteca (matplotlib.pyplot)
# As demais bibliotecas sao essenciais
import numpy
import matplotlib.pyplot as plt
import csv

class Perceptron:
    # 1 - Construtor recebe:
    # OBRIGATORIAS
    # path = caminho do arquivo
    # number_features = numero de colunas que serao utilizadas
    #
    # NAO OBRIGATORIAS
    # col1 = começo da coluna do dataset
    # col2 = fim da coluna do dataset (OBS:Se col1 e col2 nao forem declaradas, serao lidos todas as colunas, porem se um for declarado, a delcaracao do outro tambem sera precisa)
    # learning_rate = taxa de aprendizado
    # number_epoch = numero limite de epocas
    # bias = limiar de ativacao

    def __init__(self, path, number_features, col1=0, col2=-1, learning_rate=0.01, number_epoch=2000, bias=1):
        self.data_head, self.class_types, self.data_results, self.data_training = self.__read_file(path, col1, col2)
        self.learning_rate = learning_rate
        self.number_epoch = number_epoch
        self.bias = bias
        # 1.1 - Gerar os valor dos pesos aleatoreamente
        self.__w = numpy.random.uniform(0, 1, number_features)

    # 2 - Ler o arquivo csv (obs: a classe das saidas devera estar na ultima coluna)
    def __read_file(self, path, col1, col2):
        try:
            with open(path) as arquivo:
                dataset = numpy.array(list(csv.reader(arquivo)))

                # 2.1 - Pegar as saidas possiveis contidas no dataset
                try:  # Tenta pegar as saidas como inteiros
                    class_types = numpy.array(list(set(map(int, dataset[1:, -1]))))
                except:  # Tenta pegar as saidas como string se nao forem numericos
                    class_types = numpy.array(list(set(dataset[1:, -1])))

                class_types = numpy.sort(class_types)
                data_head = dataset[0]
                if col1 == 0 and col2 == -1:
                    data_training = numpy.zeros((len(dataset) - 1, len(dataset[0]) - 1))
                    column_begin = 0
                    column_end = -1
                else:
                    if col2 == -1:
                        raise ValueError('FIM DA COLUNA NAO DECALRADO')
                    data_training = numpy.zeros((len(dataset) - 1, (col2-col1+1)))
                    column_begin = col1
                    column_end = col2 + 1
                data_results = numpy.empty(len(dataset) - 1)

                # 2.2 - Inserir todos os valores na variavel (data_training) e as saidas na variavel (data_results)
                for i in range(1, len(dataset)):
                    data_training[i - 1] = dataset[i][column_begin:column_end]
                    if str(class_types[0]) == dataset[i][-1]:
                        data_results[i - 1] = -1
                    if str(class_types[1]) == dataset[i][-1]:
                        data_results[i - 1] = 1

            return data_head, class_types, data_results, data_training
        except Exception as e:
            print("ERRO AO LER O ARQUIVO")
            print(e)

    # 3 - Funçao de ativaçao
    def __verify(self, number):
        if number >= 0.0:  # Retornar 1 se o valor for maior ou igual a zero
            return 1
        else:  # Retornar -1 se o valor for menor que 0
            return -1

    # 4 - Funçao para começar o treino
    def start_train(self):
        # 4.1 - Inserir o bias no dataset dos valores e no dos pesos
        self.data_training = numpy.insert(self.data_training[:, ], len(self.data_training[0]), self.bias, axis=1)
        self.__w = numpy.insert(self.__w, len(self.__w), self.bias)

        # 4.2 - Rodar o treinamento dentro do limite das epocas inseridas no construtor
        for i in range(self.number_epoch):
            print("Interaçao: {}".format(i))
            have_error = False   # Variavel que armazena se o existe um erro entre a saida calculada e a saida correta

            # 4.3 - Percorrer todas as linhas do dataset
            for j in range(len(self.data_training)):
                # 4.4 - Fazer o produto escalar dos valores da linha atual do dataset com o vetor que armazena os pesos
                result = round(self.__w.dot(self.data_training[j]), 4)

                # 4.5 - Verificar se a saida do valor calculado e igual a da desejada
                if self.__verify(result) != self.data_results[j]:
                    have_error = True
                    # 4.5.1 - Realiza os ajuste nos pesos
                    error = self.data_results[j] - self.__verify(result)
                    self.__w += (self.learning_rate * error * self.data_training[j])

            # 4.6 - Se nao houver erros, o treinamento e terminado
            if not have_error:
                print("Fim das interacoes: {}".format(i))
                break

    # 5 - Funçao para prever a qual a classe de determinados valores
    def predict(self, values):  # Pode receber um vetor, ou uma matriz
        # 5.1 - Verificar a dimençao da matriz
        if numpy.ndim(values) == 1:
            # 5.2 - Inserir o bias
            values = numpy.insert(values, len(values), 1)
            # 5.3 - Fazer o produto escalar e verificar a saida desse valor
            prediction = self.__verify(self.__w.dot(values))
            return prediction
        else:
            values = numpy.insert(values[:, ], len(values[0]), 1, axis=1)
            prediction = [self.__verify(self.__w.dot(x)) for x in values]
            return prediction

    # 6 - Funcao para plotar um grafico contendo os dois primeiros valores do dataset em plano bidimensional
    def plot(self):
        plt.scatter(self.data_training[:, 0], self.data_training[:, 1], cmap="BuGn_r", c=(0.4*self.data_results))
        title = "Class "+str(self.class_types[0])+" x "+"Class "+str(self.class_types[1])
        plt.title(title)
        plt.xlabel(self.data_head[0])
        plt.ylabel(self.data_head[1])
        # 6.1 - Verificar o numero de pesos para saber se o treino foi realizado em um dataset bidimensional
        # Se sim, plotar a reta que separa as classes
        if(len(self.__w) == 3):
            if self.__w[0] < 0 and self.__w[1] < 0:
                x = [(((min(self.data_training[0:, 1])*self.__w[1])+self.__w[-1]) / -self.__w[0]), min(self.data_training[0:, 0])]
                y = [min(self.data_training[0:, 1]), (((min(self.data_training[0:, 0]) * self.__w[0]) + self.__w[-1]) / -self.__w[1])]
                plt.plot(x, y, color='red')
            elif self.__w[0] < 0 and self.__w[1] > 0:
                x = [(((min(self.data_training[0:, 1]) * self.__w[1]) + self.__w[-1]) / -self.__w[0]), max(self.data_training[0:, 0])]
                y = [min(self.data_training[0:, 1]), (((max(self.data_training[0:, 0]) * self.__w[0]) + self.__w[-1]) / -self.__w[0:][1])]
                plt.plot(x, y, color='red')
            elif self.__w[0] > 0 and self.__w[1] > 0:
                x = [(((max(self.data_training[0:, 1]) * self.__w[1]) + self.__w[-1]) / -self.__w[0]), max(self.data_training[0:, 0])]
                y = [max(self.data_training[0:, 1]), (((max(self.data_training[0:, 0]) * self.__w[0]) + self.__w[-1]) / -self.__w[1])]
                plt.plot(x, y, color='red')
            elif self.__w[0] > 0 and self.__w[1] < 0:
                x = [(((max(self.data_training[0:, 1]) * self.__w[1]) + self.__w[-1]) / -self.__w[0]), min(self.data_training[0:, 0])]
                y = [max(self.data_training[0:, 1]), (((min(self.data_training[0:, 0]) * self.__w[0]) + self.__w[-1]) / -self.__w[1])]
                plt.plot(x, y, color='red')

        plt.plot([0, 0], [-min(self.data_training[0:, 1]), min(self.data_training[0:, 1])], color="blue")
        plt.plot([0, 0], [-max(self.data_training[0:, 1]), max(self.data_training[0:, 1])], color="blue")
        plt.plot([-min(self.data_training[0:, 0]), min(self.data_training[0:, 0])], [0, 0], color="blue")
        plt.plot([-max(self.data_training[0:, 0]), max(self.data_training[0:, 0])], [0, 0], color="blue")
        plt.show()

    # 7 - Funcao retornar a porcentagem de acertos de previsoes
    def showAccuracy(self):
        correct = 0
        # 7.1 - Fazer a previsao do dataset
        predictions = self.predict(self.data_training[0:, :-1])
        # 7.2 - Comparar os resultados da previsao com as saidas esperadas
        for x in range(len(self.data_results)):
            if self.data_results[x] == predictions[x]:
                correct += 1
        return float(correct) / len(self.data_results)

    # 8 - Funcao para retornar os pesos
    def getWeights(self):
        return self.__w
