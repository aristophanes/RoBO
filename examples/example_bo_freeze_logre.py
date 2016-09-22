import setup_logger
import gzip
import cPickle as pickle
from robo.acquisition.ei import EI
from robo.maximizers.cmaes import CMAES
#from robo.task.synthetic_functions.branin import Branin
#from robo.task.ml.logistic_regression import LogisticRegression
from robo.task.ml.logistic_regression_freeze import LogisticRegression
from robo.solver.freeze_thaw_bayesian_optimization3 import FreezeThawBO
#from robo.solver.freeze_thaw_bayesian_optimization4 import FreezeThawBO
from robo.models.freeze_thaw_model import FreezeThawGP
from robo.maximizers.direct import Direct
from robo.initial_design.init_random_uniform import init_random_uniform

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


dataset = load_data()

logre = LogisticRegression(train=dataset[0][0], train_targets=dataset[0][1],
	valid=dataset[1][0], valid_targets=dataset[1][1],
	test=dataset[2][0], test_targets=dataset[2][1],
	n_classes=10, W=None, b=None,
	freeze=True, save=True, n_epochs=3, show=False)




freeze_thaw_model = FreezeThawGP(hyper_configs=14)

#task = ExpDecay()

acquisition_func = EI(freeze_thaw_model, X_upper=logre.X_upper, X_lower=logre.X_lower)

maximizer = Direct(acquisition_func, logre.X_lower, logre.X_upper)

bo = FreezeThawBO(acquisition_func=acquisition_func,
                          freeze_thaw_model=freeze_thaw_model,
                          maximize_func=maximizer,
                          task=logre, init_points=2)

incumbent, incumbent_value = bo.run(15)

"""
dataset = load_data()

logre = LogisticRegression(train=dataset[0][0], train_targets=dataset[0][1],
	valid=dataset[1][0], valid_targets=dataset[1][1],
	test=dataset[2][0], test_targets=dataset[2][1],
	n_classes=10, W=None, b=None, save=True, num_epochs=3)




freeze_thaw_model = FreezeThawGP(hyper_configs=14)

acquisition_func = EI(freeze_thaw_model, X_upper=logre.X_upper, X_lower=logre.X_lower)

maximizer = Direct(acquisition_func, logre.X_lower, logre.X_upper)

bo = FreezeThawBO(acquisition_func=acquisition_func,
                          freeze_thaw_model=freeze_thaw_model,
                          maximize_func=maximizer,
                          task=logre, init_points=2)

incumbent, incumbent_value = bo.run(15)
"""