import theano
import scipy.io
import numpy

def shared_dataset(data_x, data_y, borrow=True):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    # shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
    shared_x = theano.shared(data_x)#, borrow=borrow)
    # shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
    shared_y = theano.shared(data_y)#, borrow=borrow)
    return shared_x, theano.tensor.cast(shared_y, 'int32')


def load_data(data_path='data/dataMFCC.mat', theano_shared=True):
    data_xy = scipy.io.loadmat(data_path)['data']
    data_y, _, data_x = numpy.split(data_xy, [1, 1], axis=1)
    data_y = data_y.reshape(data_y.shape[0])
    if theano_shared:
        return shared_dataset(data_x, data_y)
    else:
        return data_x, data_y

