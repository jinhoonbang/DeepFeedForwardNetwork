__author__ = 'diego'

"""
softmax layer
"""

__docformat__ = 'restructedtext en'

import numpy
import theano
import theano.tensor as T

n_symbol = 43

class SoftMax(object):


    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """

        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        # initialize the biases b as a vector of n_out 0s
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        #
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        #
        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        # self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        print("inside softMax")
        print("self.p_y_given_x.shape.eval()")
        print(self.p_y_given_x.shape.eval())
        print("break")

        self.y_pred = T.dmatrix();
        for i in range(n_symbol):
            self.y_pred = T.concatenate([self.y_pred, T.argmax(self.p_y_given_x[:,3*i:3*(i+1)])], axis=1)

        # b1 = T.argmax(self.p_y_given_x[:, 0:3], axis=1)
        # b2 = T.argmax(self.p_y_given_x[:, 3:6], axis=1)
        # b3 = T.argmax(self.p_y_given_x[:, 6:9], axis=1)
        # b4 = T.argmax(self.p_y_given_x[:, 9:12], axis=1)
        # b5 = T.argmax(self.p_y_given_x[:, 12:15], axis=1)
        # b6 = T.argmax(self.p_y_given_x[:, 15:18], axis=1)
        # b7 = T.argmax(self.p_y_given_x[:, 18:21], axis=1)
        # b8 = T.argmax(self.p_y_given_x[:, 21:24], axis=1)
        # b9 = T.argmax(self.p_y_given_x[:, 24:27], axis=1)
        # b10 = T.argmax(self.p_y_given_x[:, 27:30], axis=1)
        # b11 = T.argmax(self.p_y_given_x[:, 30:33], axis=1)
        # b12 = T.argmax(self.p_y_given_x[:, 33:36], axis=1)
        # b13 = T.argmax(self.p_y_given_x[:, 36:39], axis=1)
        # b14 = T.argmax(self.p_y_given_x[:, 39:42], axis=1)
        # b15 = T.argmax(self.p_y_given_x[:, 42:45], axis=1)
        # b16 = T.argmax(self.p_y_given_x[:, 45:48], axis=1)
        # b17 = T.argmax(self.p_y_given_x[:, 48:51], axis=1)
        # b18 = T.argmax(self.p_y_given_x[:, 51:54], axis=1)
        # b19 = T.argmax(self.p_y_given_x[:, 54:57], axis=1)
        # b20 = T.argmax(self.p_y_given_x[:, 57:60], axis=1)
        # b21 = T.argmax(self.p_y_given_x[:, 60:63], axis=1)
        # b22 = T.argmax(self.p_y_given_x[:, 63:66], axis=1)
        # b23 = T.argmax(self.p_y_given_x[:, 66:69], axis=1)
        # b24 = T.argmax(self.p_y_given_x[:, 69:72], axis=1)
        # b25 = T.argmax(self.p_y_given_x[:, 72:75], axis=1)
        # b26 = T.argmax(self.p_y_given_x[:, 75:78], axis=1)
        # b27 = T.argmax(self.p_y_given_x[:, 78:81], axis=1)
        # b28 = T.argmax(self.p_y_given_x[:, 81:84], axis=1)
        # b29 = T.argmax(self.p_y_given_x[:, 84:87], axis=1)
        # b30 = T.argmax(self.p_y_given_x[:, 87:90], axis=1)
        # b31 = T.argmax(self.p_y_given_x[:, 90:93], axis=1)
        # b32 = T.argmax(self.p_y_given_x[:, 93:96], axis=1)
        # b33 = T.argmax(self.p_y_given_x[:, 96:99], axis=1)
        # b34 = T.argmax(self.p_y_given_x[:, 99:102], axis=1)
        # b35 = T.argmax(self.p_y_given_x[:, 102:105], axis=1)
        # b36 = T.argmax(self.p_y_given_x[:, 105:108], axis=1)
        # b37 = T.argmax(self.p_y_given_x[:, 108:111], axis=1)
        # b38 = T.argmax(self.p_y_given_x[:, 111:114], axis=1)
        # b39 = T.argmax(self.p_y_given_x[:, 114:117], axis=1)
        # b40 = T.argmax(self.p_y_given_x[:, 117:120], axis=1)
        # b41 = T.argmax(self.p_y_given_x[:, 120:123], axis=1)
        # b42 = T.argmax(self.p_y_given_x[:, 123:126], axis=1)
        # b43 = T.argmax(self.p_y_given_x[:, 126:129], axis=1)
        #
        # self.y_pred = T.concatenate([b1, b2 ,b3, b4, b5, b6,b7,b8,b9,b10,b11,b12,b13,b14,b15,b16,b17,b18,b19,b20,b21,b22,b23,b24,b25,b26,b27,b28,b29,b30,b31,b32,b33,b34,b35,b36,b37,b38,b39,b40,b41,b42,b43], axis=1)

        # parameters of the model
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """

        neg_log = 0
        for i in range(0, n_symbol):
            neg_log = neg_log - T.mean(T.log(self.p_y_given_x[T.arange(y.shape[0]), (y[i]+3*i+1)])),

        return neg_log
        # return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y[:,0]])

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        # if y.ndim != self.y_pred.ndim:
        #     raise TypeError(
        #         'y should have the same shape as self.y_pred',
        #         ('y', y.type, 'y_pred', self.y_pred.type)
        #     )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            meanError = 0
            for i in range(0, n_symbol):
                meanError = meanError + T.mean(T.neq(self.y_pred[:,i]), y[:,i])
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

