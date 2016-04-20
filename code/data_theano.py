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


def load_data(data_path='data/dataMFCC.mat', theano_shared=True, split=0.8):
    data_xy = scipy.io.loadmat(data_path)['data']
    data_y, _, data_x = numpy.split(data_xy, [1, 1], axis=1)
    indx_split = int(data_x.shape[0]*split)
    train_x, _, test_x = numpy.split(data_x, [indx_split, indx_split], axis=0)
    train_y, _, test_y = numpy.split(data_y, [indx_split, indx_split], axis=0)
	
    valid_split = int(train_x.shape[0]*0.25)
    
    valid_x, _, valid_x = numpy.split(train_x, [valid_split, valid_split], axis=0)
    valid_y, _, valid_y = numpy.split(train_y, [valid_split, valid_split], axis=0)
	


    #data_y = data_y.reshape(data_y.shape[0])
    print data_x.shape, data_y.shape
    print train_x.shape, train_y.shape
    print valid_x.shape, valid_y.shape
    print test_x.shape, test_x.shape
    if theano_shared:
        return (shared_dataset(train_x, train_y), shared_dataset(valid_x,valid_y), shared_dataset(test_x, test_y)),len(set(train_y.reshape(train_y.shape[0])))
    else:
        return ((train_x, train_y),(valid_x,valid_y), (test_x, test_y)),len(set(train_y.reshape(train_y.shape[0])))
