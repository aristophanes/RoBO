import numpy as np
import theano
import theano.tensor as T
import cPickle as pickle

from robo.task.base_task import BaseTask


class LogisticRegression(BaseTask):

    def __init__(self, train, train_targets,
                 valid, valid_targets,
                 test, test_targets,
                 n_classes, freeze=False,
                 W=None, b=None,
                 save=False, file_name=None, 
                 n_epochs=5, show=True):
        '''
        Logistic Regression with early stopping. This benchmark
        consists of 4 different hyperparamters:
            Learning rate
            L2 regularisation
            Batch size
            Number of epochs
        The validation dataset is only used for early stopping. The
        error metric is the classification error of the test data.

        This benchmarks is based on the experiments conducted in the
        Spearming paper by Snoek et. al. [1]. The implementation
        is mostly copied from the Theano tutorial page:
        http://deeplearning.net/tutorial/logreg.html

        [1] J. Snoek, H. Larochelle, and R.P. Adams.
            Practical Bayesian optimization of machine learning algorithms.
            In Proc. of NIPS 12

        Parameters
        ----------
        train : (N, D) numpy array
            Training matrix where N are the number of data points
            and D are the number of features
        train_targets : (N) numpy array
            Labels for the training data
        valid : (K, D) numpy array
            Validation data
        valid_targets : (K) numpy array
            Validation labels
        test : (L, D) numpy array
            Test data
        test_targets : (L) numpy array
            Test labels
        n_classes: int
            Number of classes in the dataset
        '''

        def shared_dataset(data_xy, borrow=True):
            data_x, data_y = data_xy
            shared_x = theano.shared(np.asarray(data_x,
                                           dtype=theano.config.floatX),
                                           borrow=borrow)
            shared_y = theano.shared(np.asarray(data_y,
                                    dtype=theano.config.floatX),
                                    borrow=borrow)
            return shared_x, T.cast(shared_y, 'int32')

        self.test_set_x, self.test_set_y = shared_dataset((test,
                                                           test_targets))
        self.valid_set_x, self.valid_set_y = shared_dataset((valid,
                                                             valid_targets))
        self.train_set_x, self.train_set_y = shared_dataset((train,
                                                             train_targets))

        n_in = train.shape[1]

        if W is None:
            self.W = theano.shared(value=np.zeros((n_in, n_classes),
                dtype=theano.config.floatX),
                name='W',
                borrow=True)
        else:
            self.W = W

        if b is None:
            self.b = theano.shared(value=np.zeros((n_classes,),
                dtype=theano.config.floatX),
                name='b',
                borrow=True)
        else:
            self.b = b

        self.input = T.matrix('x')
        self.output = T.ivector('y')

        self.p_y_given_x = T.nnet.softmax(T.dot(self.input, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.params = [self.W, self.b]

        self.l2_sqr = (self.W ** 2).sum()
        self.freeze_modus=freeze
        self.n_epochs = n_epochs
        # 1 Dim Learning Rate
        # 2 Dim L2 regularization
        # 3 Dim Batch size
        # 4 Dim Number of epochs
        #X_lower = np.array([np.log(1e-10), 0.0, 20, 5])
        #X_upper = np.array([np.log(1.0), 1.0, 2000, 2000])
        X_lower = np.array([np.log(1e-10), 0.0, 400])
        X_upper = np.array([np.log(1.0), 1.0, 700])
        self.save = save
        self.file_name = file_name
        self.show = show
        super(LogisticRegression, self).__init__(X_lower, X_upper, do_scaling=False)

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

    def set_weights(self, W, b):
        self.W = W
        self.b = b

    def set_epochs(self, n_epochs):
        self.n_epochs = n_epochs

    def set_save_modus(self, is_old=True, file_old=None, file_new=None):
        self.save_old = is_old
        if self.save_old:
            self.file_name = file_old
        else:
            self.file_name = file_new


    def objective_function(self, x):

        learning_rate = np.float32(np.exp(x[0]))
        #print 'learning_rate: ', learning_rate
        l2_reg = np.float32(x[1])
        #print 'l2_reg: ', l2_reg
        batch_size = np.int32(x[2])
        #print 'batch_size: ', batch_size
        #n_epochs = np.int32(x[0, 3])
        n_epochs=self.n_epochs
        curve = np.zeros(n_epochs)

        n_train_batches = self.train_set_x.get_value(borrow=True).shape[0] // batch_size
        n_valid_batches = self.valid_set_x.get_value(borrow=True).shape[0] // batch_size
        n_test_batches = self.test_set_x.get_value(borrow=True).shape[0] // batch_size

        index = T.lscalar()  # index to a [mini]batch

        cost = self.negative_log_likelihood(self.output) + l2_reg * self.l2_sqr

        test_model = theano.function(inputs=[index],
                outputs=self.errors(self.output),
                givens={self.input: self.test_set_x[index * batch_size: \
                                                    (index + 1) * batch_size],
                        self.output: self.test_set_y[index * batch_size: \
                                                     (index + 1) * batch_size]
                        })

        validate_model = theano.function(
                        inputs=[index],
                        outputs=self.errors(self.output),
                        givens={self.input: self.valid_set_x[index * batch_size:\
                                                    (index + 1) * batch_size],
                                self.output: self.valid_set_y[index * batch_size:\
                                                    (index + 1) * batch_size]
                        })

        g_W = T.grad(cost=cost, wrt=self.W, disconnected_inputs='warn')
        g_b = T.grad(cost=cost, wrt=self.b, disconnected_inputs='warn')

        updates = [(self.W, self.W - learning_rate * g_W),
               (self.b, self.b - learning_rate * g_b)]

        train_model = theano.function(
                    inputs=[index],
                    outputs=cost,
                    updates=updates,
                    givens={
                        self.input: self.train_set_x[index * batch_size: \
                                                     (index + 1) * batch_size],
                        self.output: self.train_set_y[index * batch_size: \
                                                      (index + 1) * batch_size]
                    }
                )
        #patience = 5000
        #patience_increase = 2
        #improvement_threshold = 0.995
        #validation_frequency = min(n_train_batches, patience // 2)

        best_validation_loss = np.inf
        test_score = 0.

        #done_looping = False
        epoch = 0
        while (epoch < n_epochs):# and (not done_looping):
            epoch = epoch + 1
            best_validation_batch = np.inf
            for minibatch_index in range(n_train_batches):

                minibatch_avg_cost = train_model(minibatch_index)
                # iteration number
                it = (epoch - 1) * n_train_batches + minibatch_index

                #if (it + 1) % validation_frequency == 0:
                    # compute zero-one loss on validation set
                validation_losses = [validate_model(i)
                                    for i in range(n_valid_batches)]
                this_validation_loss = np.mean(validation_losses)
                #for the freeze-thaw modell we need to save each one of these here
                #print 'epoch: ', epoch

                if self.show:
                    print('epoch %i, minibatch %i/%i, validation error %f %%' %
                        (epoch, minibatch_index + 1, n_train_batches,
                            this_validation_loss * 100.))

                    # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                        #improve patience if loss improvement is good enough
                        #if this_validation_loss < best_validation_loss *  \
                        #   improvement_threshold:
                        #    patience = max(patience, it * patience_increase)

                    best_validation_loss = this_validation_loss
                        # test it on the test set

                    test_losses = [test_model(i)
                                       for i in range(n_test_batches)]
                    test_score = np.mean(test_losses)

                    if self.show:
                        print(('epoch %i, minibatch %i/%i, test error of'
                            ' best model %f %%') % (epoch,
                            minibatch_index + 1,
                            n_train_batches,
                            test_score * 100.))
                
                if this_validation_loss < best_validation_batch:
                    best_validation_batch = this_validation_loss
                    curve[epoch-1] = best_validation_batch

            

                #if patience <= it:
                #    done_looping = True
                #    break

        weights_final = (self.W, self.b)

        if self.save:
            with open(self.file_name, 'wb') as f:
                pickle.dump(weights_final, f)        

        print(('Optimization complete with best validation score of %f %%,'
            'with test performance %f %%') % (best_validation_loss * 100.,
                                              test_score * 100.))

        self.test_error = np.array([[test_score]])

        if not self.freeze_modus:
            return np.array([[best_validation_loss]])
        else:
            return curve
        
    def objective_function_test(self, x):        
        self.objective_function(x)
        return self.test_error
