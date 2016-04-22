import numpy as np
from sklearn import mixture
from data_theano import load_data
from multiprocessing import Lock, Process, Value


class GMM:

    def __init__(self):
        self.train, self.valid, self.test = load_data(theano_shared=False)
        self.train_x, self.train_y = self.train
        self.test_x, self.test_y = self.test
        self.valid_x, self.valid_y = self.valid

        self.train_y = self.train_y.reshape(self.train_y.shape[0])
        self.test_y = self.test_y.reshape(self.test_y.shape[0])

        self.classes_number = len(set(self.train_y))

        self.model = mixture.GMM(n_components=self.classes_number)
        # self.accurate = Value('i', 0)
        self.accurate = 0

    def fit_data(self):
        self.model.fit(self.train_x, self.train_y)

    def test_accuracy(self, st, en, thread_number, var_x, var_y):
        for indx in range(st, en):
            data_pt = var_x[indx]
            temp = self.model.predict(data_pt.reshape(1, -1))[0]

            # self.lck.acquire()
            if temp == var_y[indx]:
                print self.accurate
                # self.accurate.value += + 1
                self.accurate += + 1
                print self.accurate, indx,
                print 'Thread: {}'.format(thread_number)
            # else:
                # print 'Thread: {} {}-{}'.format(thread_number, temp, var_y[indx])
                
            # self.lck.release()

    def testing(self, data_x, data_y):
        self.accurate.value = 0
        processes = []
        ranges = range(0, data_y.shape[0], data_y.shape[0] // 4)
        for x in range(len(ranges) - 1):
            p = Process(target=self.test_accuracy, args=(ranges[x], ranges[x + 1], x, data_x, data_y))
            p.start()
            processes.append(p)

        _ = map(lambda x: x.join(), processes)

        self.accuracy_percentage = self.accurate.value * 100. / data_y.shape[0]
        print 'Accuracy: {}'.format(self.accuracy_percentage)
        print 'Accurate: {}/{}'.format(self.accurate.value, data_y.shape[0])

gmm_obj = GMM()
print 'Data Loaded'
gmm_obj.fit_data()
print 'Model Fitting Done'
gmm_obj.test_accuracy(0, gmm_obj.test_y.shape[0], 0, gmm_obj.test_x, gmm_obj.test_y)
accuracy_percentage = gmm_obj.accurate * 100. / gmm_obj.test_y.shape[0]
print 'Accuracy: {}'.format(accuracy_percentage)
print 'Accurate: {}/{}'.format(gmm_obj.accurate, gmm_obj.test_y.shape[0])
# gmm_obj.testing(gmm_obj.test_x, gmm_obj.test_y)
# gmm_obj.testing(gmm_obj.valid_x, gmm_obj.valid_y)
# gmm_obj.testing(gmm_obj.train_x, gmm_obj.train_y)
