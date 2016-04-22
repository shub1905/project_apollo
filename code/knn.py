'''
K nearest neighbours
'''

from sklearn.neighbors import KNeighborsClassifier
from data_theano import load_data
import numpy
from multiprocessing import Lock, Process, Value


class KNN(object):

    def __init__(self):
        (self.train, self.valid, self.test), _ = load_data(data_path='data/mfcc_{}.npy', theano_shared=False)
        self.train_x, self.train_y = self.train
        self.test_x, self.test_y = self.test
        self.valid_x, self.valid_y = self.valid

        self.train_y = self.train_y.reshape(self.train_y.shape[0])
        self.test_y = self.test_y.reshape(self.test_y.shape[0])

        self.accurate = Value('i', 0)
        self.lck = Lock()
        # self.unique_artists()
        # self.centers = len(self.artists_map.keys())
        self.neigh = KNeighborsClassifier(weights='distance', n_jobs=-1, p=1)

    def unique_artists(self):
        self.artists_map = {}
        for artist in self.data_y:
            temp = self.artists_map.get(artist, 0)
            self.artists_map[artist] = temp + 1

    def fit_data(self):
        self.neigh.fit(self.train_x, self.train_y)

    def test_accuracy(self, st, en, thread_number, var_x, var_y):
        for indx in range(st, en):
            data_pt = var_x[indx]
            temp = self.neigh.predict(data_pt.reshape(1, -1))[0]

            if temp == var_y[indx]:
                self.lck.acquire()
                # print self.accurate.value,
                self.accurate.value += + 1
                # print self.accurate.value, indx,
                # print 'Thread: {}'.format(thread_number)
                self.lck.release()

    def testing(self, data_x, data_y):
        self.accurate.value = 0
        processes = []
        ranges = range(0, data_y.shape[0], data_y.shape[0] // 10)
        for x in range(len(ranges) - 1):
            p = Process(target=self.test_accuracy, args=(ranges[x], ranges[x + 1], x, data_x, data_y))
            p.start()
            processes.append(p)

        _ = map(lambda x: x.join(), processes)

        self.accuracy_percentage = self.accurate.value * 100. / data_y.shape[0]
        print 'Accuracy: {}'.format(self.accuracy_percentage)
        print 'Accurate: {}/{}'.format(self.accurate.value, data_y.shape[0])

knn_obj = KNN()
knn_obj.fit_data()
knn_obj.testing(knn_obj.test_x, knn_obj.test_y)
knn_obj.testing(knn_obj.valid_x, knn_obj.valid_y)
knn_obj.testing(knn_obj.train_x, knn_obj.train_y)
