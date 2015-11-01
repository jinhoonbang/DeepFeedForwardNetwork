#Jin Hoon Bang

"""
deep feed forward network for Futures from CBOE Futures Exchange
"""
import theano
import theano.tensor as T
import numpy
import glob
import pandas as pd
import numpy as np
from src.algorithms.SGD4FFN import SGD4FFN as SGD4FFN
import sys

theano.config.exception_verbosity = 'high'
theano.config.optimizer = 'fast_compile'

#log = open('DFFN_100epoch.log', 'w')
#sys.stdout = log

params = dict(
    dataset = glob.glob('/home/jbang/data/smallHybrid/*'),
    hiddenLayers = [1000, 900, 800],
    n_in = 0, #chosen after data is loaded
    n_out = 129, # number of classes
    n_row = 1000,
    batch_size = 20,
    n_epochs = 10,
    # with_projection = True, # applicable only with actOptimization
    # model = "plain" # actChoice or plain or actOptimization
)

def preprocessData(path):
    files = []
    for file in path:
        files.append(file)
    files.sort()

    dfLabel = pd.DataFrame(dtype='float64')
    dfFeature = pd.DataFrame(dtype='float64')

    for file in files:
        binary = np.fromfile(file, dtype='float64')
        numRow=binary[0]
        numCol=binary[1]
        binary=np.delete(binary,[0,1])
        binary=binary.reshape((numRow,numCol))

        tempLabel=pd.DataFrame(binary[:,0])
        tempFeature=pd.DataFrame(binary[:,1:])
        dfLabel=pd.concat([dfLabel, tempLabel],axis=1)
        dfFeature=pd.concat([dfFeature, tempFeature],axis=1)

    label = dfLabel.tail(params['n_row']).as_matrix()
    label = label+1
    print(label)
    feature = dfFeature.tail(params['n_row']).as_matrix()

    label = label.astype('int32')
    feature = feature.astype('float64')

    return feature, label

def trainTestSplit(feature, label):
    n_train = 0.6 * params['n_row']
    n_valid = 0.2 * params['n_row']
    n_test = 0.2 * params['n_row']

    x_train = feature[:n_train]
    y_train = label[:n_train]

    x_valid = feature[n_train: n_train + n_valid]
    y_valid = label[n_train: n_train + n_valid]

    x_test = feature[n_train + n_valid:]
    y_test = label[n_train + n_valid:]

    train_set = (x_train, y_train)
    valid_set = (x_valid, y_valid)
    test_set = (x_test, y_test)

    return train_set, valid_set, test_set

def load_data(dataset):
    feature, label = preprocessData(dataset)
    params['n_in'] = feature.shape[1]
    train_set, valid_set, test_set = trainTestSplit(feature, label)

    #############
    # LOAD DATA #
    #############

    print('... loading data')

    #train_set, valid_set, test_set format: tuple(input, target)
    #input and target are both numpy.ndarray of 2 dimensions (a matrix)
    #each row correspond to an example.

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy

        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)


        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval

if __name__ == '__main__':
    datasets = load_data(params['dataset'])
    SGD4FFN(datasets,params['hiddenLayers'],params['n_in'],params['n_out'],n_epochs=params['n_epochs'])

#log.close()
