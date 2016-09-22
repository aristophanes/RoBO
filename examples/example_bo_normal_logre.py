import setup_logger
import gzip
import cPickle as pickle
import numpy as np
import GPy
import george
from robo.models.gpy_model import GPyModel
from robo.models.gaussian_process_mcmc import GaussianProcessMCMC
from robo.acquisition.ei import EI
from robo.maximizers.cmaes import CMAES
from robo.maximizers.direct import Direct
#from robo.task.synthetic_functions.branin import Branin
#from robo.task.ml.logistic_regression import LogisticRegression
from robo.task.ml.lasagne_logrg_task import LogisticRegression
#from robo.task.ml.logistic_regression_freeze import LogisticRegression
from robo.solver.bayesian_optimization import BayesianOptimization
from robo.initial_design.init_random_uniform import init_random_uniform
from robo.priors.default_priors import DefaultPrior
from robo.acquisition.integrated_acquisition import IntegratedAcquisition

"""
def load_data(dataset='mnist.pkl.gz'):
	''' Loads the dataset

	:type dataset: string
	:param dataset: the path to the dataset (here MNIST)
	'''
	with gzip.open(dataset, 'rb') as f:
		try:
			train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
		except:
			train_set, valid_set, test_set = pickle.load(f)

	return train_set, valid_set, test_set
"""
def load_dataset():

    def load_mnist_images(filename):
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        # The inputs are vectors now, we reshape them to monochrome 2D images,
        # following the shape convention: (examples, channels, rows, columns)
        data = data.reshape(-1, 1, 28, 28)
        # The inputs come as bytes, we convert them to float32 in range [0,1].
        # (Actually to range [0, 255/256], for compatibility to the version
        # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
        return data / np.float32(256)

    def load_mnist_labels(filename):
        # Read the labels in Yann LeCun's binary format.
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        # The labels are vectors of integers now, that's exactly what we want.
        return data

    # We can now download and read the training and test set images and labels.
    X_train = load_mnist_images('train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

    # We reserve the last 10000 training examples for validation.
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    # We just return all the arrays in order, as expected in main().
    # (It doesn't matter how we do this as long as we can read them again.)
    return X_train, y_train, X_val, y_val, X_test, y_test


X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()


"""
logre = LogisticRegression(train=dataset[0][0], train_targets=dataset[0][1],
	valid=dataset[1][0], valid_targets=dataset[1][1],
	test=dataset[2][0], test_targets=dataset[2][1],
	n_classes=10, W=None, b=None,
	freeze=True, save=True, file_name='model_test.pkl')



init = init_random_uniform(logre.X_lower, logre.X_upper, 1)


result = logre.objective_function(x=init)

print 'result: ', result
"""
"""
W,b = pickle.load(open('model_test.pkl'))

logre = LogisticRegression(train=dataset[0][0], train_targets=dataset[0][1],
	valid=dataset[1][0], valid_targets=dataset[1][1],
	test=dataset[2][0], test_targets=dataset[2][1],
	n_classes=10, W=W, b=b,
	freeze=True, save=True, file_name='model_test.pkl')

init = init_random_uniform(logre.X_lower, logre.X_upper, 1)


result = logre.objective_function(x=init)

print 'result: ', result
print

W,b = pickle.load(open('model_test.pkl'))

logre = LogisticRegression(train=dataset[0][0], train_targets=dataset[0][1],
	valid=dataset[1][0], valid_targets=dataset[1][1],
	test=dataset[2][0], test_targets=dataset[2][1],
	n_classes=10, W=W, b=b,
	freeze=True, save=True, file_name='model_test.pkl')

#init = init_random_uniform(logre.X_lower, logre.X_upper, 1)


result = logre.objective_function(x=init)

print 'result: ', result
print

W,b = pickle.load(open('model_test.pkl'))

logre = LogisticRegression(train=dataset[0][0], train_targets=dataset[0][1],
	valid=dataset[1][0], valid_targets=dataset[1][1],
	test=dataset[2][0], test_targets=dataset[2][1],
	n_classes=10, W=W, b=b,
	freeze=True, save=True, file_name='model_test.pkl')

#init = init_random_uniform(logre.X_lower, logre.X_upper, 1)


result = logre.objective_function(x=init)

print 'result: ', result
print

W,b = pickle.load(open('model_test.pkl'))

logre = LogisticRegression(train=dataset[0][0], train_targets=dataset[0][1],
	valid=dataset[1][0], valid_targets=dataset[1][1],
	test=dataset[2][0], test_targets=dataset[2][1],
	n_classes=10, W=W, b=b,
	freeze=True, save=True, file_name='model_test.pkl')

#init = init_random_uniform(logre.X_lower, logre.X_upper, 1)


result = logre.objective_function(x=init)

print 'result: ', result
print

W,b = pickle.load(open('model_test.pkl'))

logre = LogisticRegression(train=dataset[0][0], train_targets=dataset[0][1],
	valid=dataset[1][0], valid_targets=dataset[1][1],
	test=dataset[2][0], test_targets=dataset[2][1],
	n_classes=10, W=W, b=b,
	freeze=True, save=True, file_name='model_test.pkl')

#init = init_random_uniform(logre.X_lower, logre.X_upper, 1)


result = logre.objective_function(x=init)

print 'result: ', result
print

W,b = pickle.load(open('model_test.pkl'))

logre = LogisticRegression(train=dataset[0][0], train_targets=dataset[0][1],
	valid=dataset[1][0], valid_targets=dataset[1][1],
	test=dataset[2][0], test_targets=dataset[2][1],
	n_classes=10, W=W, b=b,
	freeze=True, save=True, file_name='model_test.pkl')

#init = init_random_uniform(logre.X_lower, logre.X_upper, 1)


result = logre.objective_function(x=init)

print 'result: ', result
print

W,b = pickle.load(open('model_test.pkl'))

logre = LogisticRegression(train=dataset[0][0], train_targets=dataset[0][1],
	valid=dataset[1][0], valid_targets=dataset[1][1],
	test=dataset[2][0], test_targets=dataset[2][1],
	n_classes=10, W=W, b=b,
	freeze=True, save=True, file_name='model_test.pkl')

#init = init_random_uniform(logre.X_lower, logre.X_upper, 1)


result = logre.objective_function(x=init)

print 'result: ', result
print

W,b = pickle.load(open('model_test.pkl'))

logre = LogisticRegression(train=dataset[0][0], train_targets=dataset[0][1],
	valid=dataset[1][0], valid_targets=dataset[1][1],
	test=dataset[2][0], test_targets=dataset[2][1],
	n_classes=10, W=W, b=b,
	freeze=True, save=True, file_name='model_test.pkl')

#init = init_random_uniform(logre.X_lower, logre.X_upper, 1)


result = logre.objective_function(x=init)

print 'result: ', result
print
"""

"""

logre = LogisticRegression(train=dataset[0][0], train_targets=dataset[0][1],
	valid=dataset[1][0], valid_targets=dataset[1][1],
	test=dataset[2][0], test_targets=dataset[2][1],
	n_classes=10)

kernel = GPy.kern.Matern52(input_dim=logre.n_dims)

model = GPyModel(kernel)

acquisition_func = EI(model,
                     X_upper=logre.X_upper,
                     X_lower=logre.X_lower,
                     par=0.1)

maximizer = CMAES(acquisition_func, logre.X_lower, logre.X_upper)

bo = BayesianOptimization(acquisition_func=acquisition_func,
                          model=model,
                          maximize_func=maximizer,
                          task=logre)

bo.run(10)
"""

"""
#logre = LogisticRegression(train=dataset[0][0], train_targets=dataset[0][1],
#	valid=dataset[1][0], valid_targets=dataset[1][1],
#	test=dataset[2][0], test_targets=dataset[2][1],
#	n_classes=10, num_epochs=10)
logre = LogisticRegression(train=X_train, train_targets=y_train,
	valid=X_val, valid_targets=y_val,
	test=X_test, test_targets=y_test,
	n_classes=10, num_epochs=3)

kernel = GPy.kern.Matern52(input_dim=logre.n_dims)

model = GPyModel(kernel)

acquisition_func = EI(model,
                     X_upper=logre.X_upper,
                     X_lower=logre.X_lower,
                     par=0.1)

maximizer = CMAES(acquisition_func, logre.X_lower, logre.X_upper)

bo = BayesianOptimization(acquisition_func=acquisition_func,
                          model=model,
                          maximize_func=maximizer,
                          task=logre)

bo.run(7)
"""

"""
logre = LogisticRegression(train=dataset[0][0], train_targets=dataset[0][1],
	valid=dataset[1][0], valid_targets=dataset[1][1],
	test=dataset[2][0], test_targets=dataset[2][1],
	n_classes=10, num_epochs=10)

noise = 1.0
cov_amp = 2
exp_kernel = george.kernels.Matern52Kernel([1.0, 1.0], ndim=logre.n_dims)
kernel = cov_amp * exp_kernel

prior = DefaultPrior(len(kernel) + 1)
model = GaussianProcessMCMC(kernel, prior=prior,
                            chain_length=100, burnin_steps=200, n_hypers=20)

ei = EI(model, logre.X_lower, logre.X_upper)
acquisition_func = IntegratedAcquisition(model, ei, logre.X_lower, logre.X_upper)

maximizer = Direct(acquisition_func, logre.X_lower, logre.X_upper)

bo = BayesianOptimization(acquisition_func=acquisition_func,
                          model=model,
                          maximize_func=maximizer,
                          task=task)

print bo.run(10)
"""

logre = LogisticRegression(train=X_train, train_targets=y_train,
	valid=X_val, valid_targets=y_val,
	test=X_test, test_targets=y_test,
	n_classes=10, num_epochs=3)

noise = 1.0
cov_amp = 2
#exp_kernel = george.kernels.Matern52Kernel([1.0, 1.0], ndim=logre.n_dims)
lengths = [1.0 for i in xrange(logre.n_dims)]
exp_kernel = george.kernels.Matern52Kernel(lengths, ndim=logre.n_dims)
kernel = cov_amp * exp_kernel

prior = DefaultPrior(len(kernel) + 1)
model = GaussianProcessMCMC(kernel, prior=prior,
                            chain_length=100, burnin_steps=200, n_hypers=20)

ei = EI(model, logre.X_lower, logre.X_upper)
acquisition_func = IntegratedAcquisition(model, ei, logre.X_lower, logre.X_upper)

maximizer = Direct(acquisition_func, logre.X_lower, logre.X_upper)

bo = BayesianOptimization(acquisition_func=acquisition_func,
                          model=model,
                          maximize_func=maximizer,
                          task=logre)

print bo.run(7)