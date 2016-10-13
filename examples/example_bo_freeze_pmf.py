import setup_logger
import gzip
import cPickle as pickle
from robo.acquisition.ei import EI
from robo.maximizers.cmaes import CMAES
import numpy as np
#from robo.task.synthetic_functions.branin import Branin
#from robo.task.ml.logistic_regression import LogisticRegression
#from robo.task.ml.logistic_regression_freeze import LogisticRegression
#from robo.task.ml.lasagne_logrg_task_freeze import LogisticRegression
from robo.task.ml.pmf_task_freeze import PMF
from robo.solver.freeze_thaw_bayesian_optimization4 import FreezeThawBO
#from robo.solver.freeze_thaw_bayesian_optimization3 import FreezeThawBO
from robo.models.freeze_thaw_model import FreezeThawGP
#from robo.models.freeze_thaw_model_ec import FreezeThawGP
from robo.maximizers.direct import Direct
from robo.initial_design.init_random_uniform import init_random_uniform

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



pmf_task = PMF(n_classes=10, num_epochs=3,
  iteration_nr=3)


#logre.X_upper = np.array([np.log(1e-1), 1.0, 2000, 0.75, 20, 10])

freeze_thaw_model = FreezeThawGP(hyper_configs=14)
#freeze_thaw_model = FreezeThawGP(hyper_configs=14, economically=False)

#task = ExpDecay()

acquisition_func = EI(freeze_thaw_model, X_upper=pmf_task.X_upper, X_lower=pmf_task.X_lower)

maximizer = Direct(acquisition_func, pmf_task.X_lower, pmf_task.X_upper)

bo = FreezeThawBO(acquisition_func=acquisition_func,
                          freeze_thaw_model=freeze_thaw_model,
                          maximize_func=maximizer,
                          task=pmf_task, init_points=2)

"""
bo = FreezeThawBO(acquisition_func=acquisition_func,
                          freeze_thaw_model=freeze_thaw_model,
                          maximize_func=maximizer,
                          task=logre, init_points=10,
                          max_epochs=500, stop_epochs=True)
"""
incumbent, incumbent_value = bo.run(5)