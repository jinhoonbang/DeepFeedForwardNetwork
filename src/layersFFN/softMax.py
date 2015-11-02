__author__ = 'diego'

"""
softmax layer
"""

__docformat__ = 'restructedtext en'

import numpy
import theano
import theano.tensor as T
from theano.printing import debugprint
from theano import pp

n_symbol = 43
n_rows = 50000

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

        # dotproduct = T.dot(input, self.W) + self.b

        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        p1 = self.p_y_given_x[:, 0:3]/T.sum(self.p_y_given_x[:,0:3], axis=1, keepdims=True)
        p2 = self.p_y_given_x[:, 3:6]/T.sum(self.p_y_given_x[:,3:6], axis=1, keepdims=True)
        p3 = self.p_y_given_x[:, 6:9]/T.sum(self.p_y_given_x[:,6:9], axis=1, keepdims=True)
        p4 = self.p_y_given_x[:, 9:12]/T.sum(self.p_y_given_x[:,9:12], axis=1, keepdims=True)
        p5 = self.p_y_given_x[:, 12:15]/T.sum(self.p_y_given_x[:,12:15], axis=1, keepdims=True)
        p6 = self.p_y_given_x[:, 15:18]/T.sum(self.p_y_given_x[:,15:18], axis=1, keepdims=True)
        p7 = self.p_y_given_x[:, 18:21]/T.sum(self.p_y_given_x[:,18:21], axis=1, keepdims=True)
        p8 = self.p_y_given_x[:, 21:24]/T.sum(self.p_y_given_x[:,21:24], axis=1, keepdims=True)
        p9 = self.p_y_given_x[:, 24:27]/T.sum(self.p_y_given_x[:,24:27], axis=1, keepdims=True)
        p10 = self.p_y_given_x[:, 27:30]/T.sum(self.p_y_given_x[:,27:30], axis=1, keepdims=True)
        p11 = self.p_y_given_x[:, 30:33]/T.sum(self.p_y_given_x[:,30:33], axis=1, keepdims=True)
        p12 = self.p_y_given_x[:, 33:36]/T.sum(self.p_y_given_x[:,33:36], axis=1, keepdims=True)
        p13 = self.p_y_given_x[:, 36:39]/T.sum(self.p_y_given_x[:,36:39], axis=1, keepdims=True)
        p14 = self.p_y_given_x[:, 39:42]/T.sum(self.p_y_given_x[:,39:42], axis=1, keepdims=True)
        p15 = self.p_y_given_x[:, 42:45]/T.sum(self.p_y_given_x[:,42:45], axis=1, keepdims=True)
        p16 = self.p_y_given_x[:, 45:48]/T.sum(self.p_y_given_x[:,45:48], axis=1, keepdims=True)
        p17 = self.p_y_given_x[:, 48:51]/T.sum(self.p_y_given_x[:,48:51], axis=1, keepdims=True)
        p18 = self.p_y_given_x[:, 51:54]/T.sum(self.p_y_given_x[:,51:54], axis=1, keepdims=True)
        p19 = self.p_y_given_x[:, 54:57]/T.sum(self.p_y_given_x[:,54:57], axis=1, keepdims=True)
        p20 = self.p_y_given_x[:, 57:60]/T.sum(self.p_y_given_x[:,57:60], axis=1, keepdims=True)
        p21 = self.p_y_given_x[:, 60:63]/T.sum(self.p_y_given_x[:,60:63], axis=1, keepdims=True)
        p22 = self.p_y_given_x[:, 63:66]/T.sum(self.p_y_given_x[:,63:66], axis=1, keepdims=True)
        p23 = self.p_y_given_x[:, 66:69]/T.sum(self.p_y_given_x[:,66:69], axis=1, keepdims=True)
        p24 = self.p_y_given_x[:, 69:72]/T.sum(self.p_y_given_x[:,69:72], axis=1, keepdims=True)
        p25 = self.p_y_given_x[:, 72:75]/T.sum(self.p_y_given_x[:,72:75], axis=1, keepdims=True)
        p26 = self.p_y_given_x[:, 75:78]/T.sum(self.p_y_given_x[:,75:78], axis=1, keepdims=True)
        p27 = self.p_y_given_x[:, 78:81]/T.sum(self.p_y_given_x[:,78:81], axis=1, keepdims=True)
        p28 = self.p_y_given_x[:, 81:84]/T.sum(self.p_y_given_x[:,81:84], axis=1, keepdims=True)
        p29 = self.p_y_given_x[:, 84:87]/T.sum(self.p_y_given_x[:,84:87], axis=1, keepdims=True)
        p30 = self.p_y_given_x[:, 87:90]/T.sum(self.p_y_given_x[:,87:90], axis=1, keepdims=True)
        p31 = self.p_y_given_x[:, 90:93]/T.sum(self.p_y_given_x[:,90:93], axis=1, keepdims=True)
        p32 = self.p_y_given_x[:, 93:96]/T.sum(self.p_y_given_x[:,93:96], axis=1, keepdims=True)
        p33 = self.p_y_given_x[:, 96:99]/T.sum(self.p_y_given_x[:,96:99], axis=1, keepdims=True)
        p34 = self.p_y_given_x[:, 99:102]/T.sum(self.p_y_given_x[:,99:102], axis=1, keepdims=True)
        p35 = self.p_y_given_x[:, 102:105]/T.sum(self.p_y_given_x[:,102:105], axis=1, keepdims=True)
        p36 = self.p_y_given_x[:, 105:108]/T.sum(self.p_y_given_x[:,105:108], axis=1, keepdims=True)
        p37 = self.p_y_given_x[:, 108:111]/T.sum(self.p_y_given_x[:,108:111], axis=1, keepdims=True)
        p38 = self.p_y_given_x[:, 111:114]/T.sum(self.p_y_given_x[:,111:114], axis=1, keepdims=True)
        p39 = self.p_y_given_x[:, 114:117]/T.sum(self.p_y_given_x[:,114:117], axis=1, keepdims=True)
        p40 = self.p_y_given_x[:, 117:120]/T.sum(self.p_y_given_x[:,117:120], axis=1, keepdims=True)
        p41 = self.p_y_given_x[:, 120:123]/T.sum(self.p_y_given_x[:,120:123], axis=1, keepdims=True)
        p42 = self.p_y_given_x[:, 123:126]/T.sum(self.p_y_given_x[:,123:126], axis=1, keepdims=True)
        p43 = self.p_y_given_x[:, 126:129]/T.sum(self.p_y_given_x[:,126:129], axis=1, keepdims=True)

        self.p_y_given_x = T.concatenate([p1, p2 ,p3, p4, p5, p6,p7,p8,p9,p10,p11,p12,p13,p14 ,p15, p16,p17,p18,p19,p20,p21,p22,p23,p24,p25,p26,p27,p28,p29,p30,p31,p32,p33,p34,p35,p36,p37,p38,p39,p40,p41,p42,p43], axis=1)



        #modify softmax so that it calculates the sum of three columns instead of all columns


        #
        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        # self.y_pred = T.argmax(self.p_y_given_x[:, 0:3], axis=1)

        # self.y_pred = theano.shared(
        #     value = numpy.zeros(
        #         (n_rows, n_symbol),
        #         dtype=theano.config.floatX
        #     ),
        #     name='y_pred',
        #     borrow=True
        # )

        # self.y_pred = T.zeros_like(y)
        # for i in range(n_symbol):
        #     self.new_y_pred = T.set_subtensor(y[:,i], T.argmax(self.p_y_given_x[:,3*i:3*(i+1)],axis=1))

        # for i in range(n_symbol):
        #     self.y_pred = T.stack([self.y_pred, T.argmax(self.p_y_given_x)[:,3*i:3*(i+1)]],axis=1)


        #Not working.
        # self.y_pred = T.imatrix()
        # for i in range(n_symbol):
        #     self.y_pred = T.concatenate([self.y_pred, T.argmax(self.p_y_given_x[:,3*i:3*(i+1)], axis=1, keepdims=True)], axis=1)


        #working
        b1 = T.argmax(self.p_y_given_x[:, 0:3], axis=1, keepdims=True)
        b2 = T.argmax(self.p_y_given_x[:, 3:6], axis=1, keepdims=True)
        b3 = T.argmax(self.p_y_given_x[:, 6:9], axis=1, keepdims=True)
        b4 = T.argmax(self.p_y_given_x[:, 9:12], axis=1, keepdims=True)
        b5 = T.argmax(self.p_y_given_x[:, 12:15], axis=1, keepdims=True)
        b6 = T.argmax(self.p_y_given_x[:, 15:18], axis=1, keepdims=True)
        b7 = T.argmax(self.p_y_given_x[:, 18:21], axis=1, keepdims=True)
        b8 = T.argmax(self.p_y_given_x[:, 21:24], axis=1, keepdims=True)
        b9 = T.argmax(self.p_y_given_x[:, 24:27], axis=1, keepdims=True)
        b10 = T.argmax(self.p_y_given_x[:, 27:30], axis=1, keepdims=True)
        b11 = T.argmax(self.p_y_given_x[:, 30:33], axis=1, keepdims=True)
        b12 = T.argmax(self.p_y_given_x[:, 33:36], axis=1, keepdims=True)
        b13 = T.argmax(self.p_y_given_x[:, 36:39], axis=1, keepdims=True)
        b14 = T.argmax(self.p_y_given_x[:, 39:42], axis=1, keepdims=True)
        b15 = T.argmax(self.p_y_given_x[:, 42:45], axis=1, keepdims=True)
        b16 = T.argmax(self.p_y_given_x[:, 45:48], axis=1, keepdims=True)
        b17 = T.argmax(self.p_y_given_x[:, 48:51], axis=1, keepdims=True)
        b18 = T.argmax(self.p_y_given_x[:, 51:54], axis=1, keepdims=True)
        b19 = T.argmax(self.p_y_given_x[:, 54:57], axis=1, keepdims=True)
        b20 = T.argmax(self.p_y_given_x[:, 57:60], axis=1, keepdims=True)
        b21 = T.argmax(self.p_y_given_x[:, 60:63], axis=1, keepdims=True)
        b22 = T.argmax(self.p_y_given_x[:, 63:66], axis=1, keepdims=True)
        b23 = T.argmax(self.p_y_given_x[:, 66:69], axis=1, keepdims=True)
        b24 = T.argmax(self.p_y_given_x[:, 69:72], axis=1, keepdims=True)
        b25 = T.argmax(self.p_y_given_x[:, 72:75], axis=1, keepdims=True)
        b26 = T.argmax(self.p_y_given_x[:, 75:78], axis=1, keepdims=True)
        b27 = T.argmax(self.p_y_given_x[:, 78:81], axis=1, keepdims=True)
        b28 = T.argmax(self.p_y_given_x[:, 81:84], axis=1, keepdims=True)
        b29 = T.argmax(self.p_y_given_x[:, 84:87], axis=1, keepdims=True)
        b30 = T.argmax(self.p_y_given_x[:, 87:90], axis=1, keepdims=True)
        b31 = T.argmax(self.p_y_given_x[:, 90:93], axis=1, keepdims=True)
        b32 = T.argmax(self.p_y_given_x[:, 93:96], axis=1, keepdims=True)
        b33 = T.argmax(self.p_y_given_x[:, 96:99], axis=1, keepdims=True)
        b34 = T.argmax(self.p_y_given_x[:, 99:102], axis=1, keepdims=True)
        b35 = T.argmax(self.p_y_given_x[:, 102:105], axis=1, keepdims=True)
        b36 = T.argmax(self.p_y_given_x[:, 105:108], axis=1, keepdims=True)
        b37 = T.argmax(self.p_y_given_x[:, 108:111], axis=1, keepdims=True)
        b38 = T.argmax(self.p_y_given_x[:, 111:114], axis=1, keepdims=True)
        b39 = T.argmax(self.p_y_given_x[:, 114:117], axis=1, keepdims=True)
        b40 = T.argmax(self.p_y_given_x[:, 117:120], axis=1, keepdims=True)
        b41 = T.argmax(self.p_y_given_x[:, 120:123], axis=1, keepdims=True)
        b42 = T.argmax(self.p_y_given_x[:, 123:126], axis=1, keepdims=True)
        b43 = T.argmax(self.p_y_given_x[:, 126:129], axis=1, keepdims=True)

        self.y_pred = T.concatenate([b1, b2 ,b3, b4, b5, b6,b7,b8,b9,b10,b11,b12,b13,b14,b15,b16,b17,b18,b19,b20,b21,b22,b23,b24,b25,b26,b27,b28,b29,b30,b31,b32,b33,b34,b35,b36,b37,b38,b39,b40,b41,b42,b43], axis=1)

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

        # neg_log = 0
        # for i in range(0, n_symbol):
        #     #yindex = T.cast(y[:,i]+3*i+1, 'int32')
        #     yindex = y[:,i]+3*i
        #     neg_log = neg_log - T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), yindex])
        # return neg_log

        m1 = -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y[:,0]+1])
        m2 = -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y[:,1]+4])
        m3 = -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y[:,2]+7])
        m4 = -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y[:,3]+10])
        m5 = -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y[:,4]+13])
        m6 = -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y[:,5]+16])
        m7 = -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y[:,6]+19])
        m8 = -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y[:,7]+22])
        m9 = -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y[:,8]+25])
        m10 = -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y[:,9]+28])
        m11 = -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y[:,10]+31])
        m12 = -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y[:,11]+34])
        m13 = -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y[:,12]+37])
        m14 = -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y[:,13]+40])
        m15 = -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y[:,14]+43])
        m16 = -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y[:,15]+46])
        m17 = -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y[:,16]+49])
        m18 = -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y[:,17]+52])
        m19 = -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y[:,18]+55])
        m20 = -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y[:,19]+58])
        m21 = -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y[:,20]+61])
        m22 = -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y[:,21]+64])
        m23 = -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y[:,22]+67])
        m24 = -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y[:,23]+70])
        m25 = -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y[:,24]+73])
        m26 = -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y[:,25]+76])
        m27 = -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y[:,26]+79])
        m28 = -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y[:,27]+82])
        m29 = -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y[:,28]+85])
        m30 = -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y[:,29]+88])
        m31 = -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y[:,30]+91])
        m32 = -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y[:,31]+94])
        m33 = -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y[:,32]+97])
        m34 = -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y[:,33]+100])
        m35 = -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y[:,34]+103])
        m36 = -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y[:,35]+106])
        m37 = -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y[:,36]+109])
        m38 = -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y[:,37]+112])
        m39 = -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y[:,38]+115])
        m40 = -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y[:,39]+118])
        m41 = -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y[:,40]+121])
        m42 = -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y[:,41]+124])
        m43 = -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y[:,42]+127])

        return (m1+m2+m3+m4+m5+m6+m7+m8+m9+m10+m11+m12+m13+m14+m15+m16+m17+m18+m19
                +m20+m21+m22+m23+m24+m25+m26+m27+m28+m29+m30+m31+m32+m33+m34+m35+m36+
                m37+m38+m39+m40+m41+m42+m43)

        # return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
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
        y = T.cast(y, 'int32')
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
            # meanError = 0
            # for i in range(0, n_symbol):
            #     meanError = meanError + T.mean(T.neq(self.y_pred[:,i], y[:,i]))
            # return meanError
        else:
            raise NotImplementedError()

