import theano
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
    shared_x = theano.shared(data_x)  # , borrow=borrow)
    # shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
    shared_y = theano.shared(data_y)  # , borrow=borrow)
    return shared_x, theano.tensor.cast(shared_y, 'int32')


def load_data(data_path='data/mfcc_{}.npy', theano_shared=True):
    test = numpy.load(data_path.format('test'))
    train = numpy.load(data_path.format('train'))
    valid = numpy.load(data_path.format('valid'))

    train_y, _, train_x = numpy.split(train, [1, 1], axis=1)
    test_y, _, test_x = numpy.split(test, [1, 1], axis=1)
    valid_y, _, valid_x = numpy.split(valid, [1, 1], axis=1)

    #data_y = data_y.reshape(data_y.shape[0])
    print train_x.shape, train_y.shape
    print valid_x.shape, valid_y.shape
    print test_x.shape, test_y.shape
    if theano_shared:
        return (shared_dataset(train_x, train_y), shared_dataset(valid_x, valid_y), shared_dataset(test_x, test_y))
    else:
        return ((train_x, train_y), (valid_x, valid_y), (test_x, test_y))