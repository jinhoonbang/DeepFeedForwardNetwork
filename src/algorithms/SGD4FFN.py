import os
import sys
import timeit
import numpy
import theano
import theano.tensor as T
import math

from src.configurations.FFN import FFN

def get_fscore(y_actual, y_pred):

    n_class = 3
    n_symbol = 43
    n_rows = 10000
    sumTP = 0
    sumTPFP = 0
    sumTPFN = 0
    for s in range(0, n_symbol):
        cur_y_actual = y_actual[:, s]
        cur_y_pred = y_pred[:, s]

        for c in range(0, n_class):
            tp = 0
            fp = 0
            fn = 0
            tn = 0
            for i in range(0, n_rows):
                if (cur_y_actual[i] == c and cur_y_pred[i] == c):
                    tp += 1
                elif (cur_y_actual[i] == c and cur_y_pred[i] != c):
                    fn += 1
                elif (cur_y_actual[i] != c and cur_y_pred[i] == c):
                    fp += 1
                elif (cur_y_actual[i] != c and cur_y_pred[i] != c):
                    tn += 1

            print(tp)
            print(fn)
            print(fp)
            print(tn)

            sumTP += tp
            sumTPFP += tp + fp
            sumTPFN += tp + fn

    pi = sumTP/sumTPFP
    p = sumTP/sumTPFN

    fscore = (2 * pi * p) / (pi + p)
    return fscore

def SGD4FFN(datasets, layers_hidden, n_in, n_out, learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=1000,
          batch_size=20):
    """
   stochastic gradient descent optimization for feed forward network

    :type dataset: string
    :param dataset:

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic gradient

    :type L1_reg: float
    :param L1_reg: L1-norm's weight when added to the cost (see
    regularization)

    :type L2_reg: float
    :param L2_reg: L2-norm's weight when added to the cost (see
    regularization)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

   """

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    print("train_set_x")
    print(train_set_x.shape.eval())
    print("train_set_y")
    print(train_set_y.shape.eval())
    print("valid_set_x")
    print(valid_set_x.shape.eval())
    print("valid_set_y")
    print(valid_set_y.shape.eval())
    print("test_set_x")
    print(test_set_x.shape.eval())
    print("test_set_y")
    print(test_set_y.shape.eval())

    # compute number of minibatches for training, validation and testing
    n_train_batches = int(train_set_x.get_value(borrow=True).shape[0] / batch_size)
    n_valid_batches = int(valid_set_x.get_value(borrow=True).shape[0] / batch_size)
    n_test_batches = int(test_set_x.get_value(borrow=True).shape[0] / batch_size)
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as matrix
    y = T.imatrix('y')  # the labels are presented as matrix

    rng = numpy.random.RandomState(1234)

    # construct the FFN class
    classifier = FFN(
        rng=rng,
        input=x,
        layers_hidden=layers_hidden,
        n_in=n_in,
        n_out=n_out,
    )

    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost = (
        classifier.negative_log_likelihood(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )

    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    predict_model = theano.function(
        inputs = [x],
        outputs = classifier.y_pred
    )

    predict_proba = theano.function(
        inputs = [x],
        outputs = classifier.p_y_given_x
    )

    # y_test = theano.function(
    #     outputs = classifier.y_pred,
    # )

    # compute the gradient of cost with respect to theta (sorted in params)
    # the resulting gradients will be stored in a list gparams
    gparams = [T.grad(cost, param) for param in classifier.params]

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs

    # given two lists of the same length, A = [a1, a2, a3, a4] and
    # B = [b1, b2, b3, b4], zip generates a list C of same size, where each
    # element is a pair formed from the two lists :
    #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
        ]

    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')

    # early-stopping parameters
    patience = 100000
    patience_increase = 30
    #patience = 1000  # look as this many examples regardless
    # patience_increase = 2  # wait this much longer when a new best is
    # found
    improvement_threshold = 0.995  # a relative improvement of this much is
    # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
    # go through this many
    # minibatche before checking the network
    # on the validation set; in this case we
    # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1

        for minibatch_index in range(n_train_batches):

            minibatch_avg_cost = train_model(minibatch_index)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print(
                    'epoch %i, minibatch %i/%i, validation error %f %% average loss %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.,
                        minibatch_avg_cost
                    )
                )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if (
                                this_validation_loss < best_validation_loss *
                                improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [test_model(i) for i
                                   in range(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %% average loss %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100., minibatch_avg_cost))

            if patience <= iter:
                done_looping = True
                break

    y_pred = predict_model(test_set_x.get_value(borrow = True))
    p_y_given_x = predict_proba(test_set_x.get_value(borrow=True))

    print(p_y_given_x)

    end_time = timeit.default_timer()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print('The code for file ' +
        os.path.split(__file__)[1] +
        ' ran for %.2fm' % ((end_time - start_time) / 60.))

    y_actual = test_set_y.eval()

    print(y_pred.shape)
    print(y_actual.shape)
    
    print("y_pred", y_pred)
    print("y_actual", y_actual)
    fscore = get_fscore(y_actual, y_pred)
    print("fscore", fscore)
    #y_pred = y_pred.tolist()
    #y_actual = y_actual.tolist()


#    microf1 = []
#    macrof1 = []
#    weightedf1 = []
#    for i in range(0, 43):
#        microf1.append(f1_score(y_actual[:,i], y_pred[:,i], average = "macro"))
#        macrof1.append(f1_score(y_actual[:,i], y_pred[:,i], average = "micro"))
#        weightedf1.append(f1_score(y_actual[:,i], y_pred[:,i], average = "weighted"))

#    microf1 = sum(microf1)/len(microf1)
#    macrof1 = sum(macrof1)/len(macrof1)
#    weightedf1 = sum(weightedf1)/len(weightedf1)

#    print("microf1", microf1)
#    print("macrof1", macrof1)
#    print("weighted1", weightedf1)

