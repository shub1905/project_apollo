'''
K nearest neighbours
'''

from sklearn.neighbors import KNeighborsClassifier
from data_theano import load_data
import numpy
from multiprocessing import Lock, Process, Value


class KNN(object):

    def __init__(self):
        self.data_x, self.data_y = load_data(theano_shared=False)
        indx_split = int(self.data_x.shape[0]*.8)
        self.train_x, _, self.test_x = numpy.split(self.data_x, [indx_split, indx_split], axis=0)
        self.train_y, _, self.test_y = numpy.split(self.data_y, [indx_split, indx_split], axis=0)

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

    def test_accuracy(self, st, en, thread_number):
        for indx in range(st, en):
            data_pt = self.test_x[indx]
            temp = self.neigh.predict(data_pt.reshape(1, -1))[0]

            # if indx%100 == 0:
            # 	self.lck.acquire()
            # 	print self.accurate
            # 	self.lck.release()

            if temp == self.test_y[indx]:
                self.lck.acquire()
                print self.accurate.value,
                self.accurate.value +=  + 1
                print self.accurate.value, indx,
                print 'Thread: {}'.format(thread_number)
                self.lck.release()

    def testing(self):
        processes = []
        ranges = range(0, self.test_y.shape[0], self.test_y.shape[0] // 10)
        for x in range(len(ranges) - 1):
            p = Process(target=self.test_accuracy,
                        args=(ranges[x], ranges[x + 1], x))
            p.start()
            processes.append(p)

        _ = map(lambda x: x.join(), processes)

        self.test_accuracy = self.accurate.value * 1.0 / self.test_y.shape[0]

knn_obj = KNN()
print knn_obj.test_y.shape
knn_obj.fit_data()
knn_obj.testing()
print knn_obj.test_accuracy * 100
print knn_obj.accurate.value, knn_obj.test_y.shape[0]
