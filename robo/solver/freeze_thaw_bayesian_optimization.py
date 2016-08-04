# encoding=utf8
__author__ = "Tulio Paiva"
__email__ = "paivat@cs.uni-freiburg.de"

import numpy as np
from copy import deepcopy
import logging
import time
from robo.initial_design.init_random_uniform import init_random_uniform
from robo.task.base_task import BaseTask
from robo.solver.base_solver import BaseSolver
from robo.incumbent.best_observation import BestObservation
from robo.models.freeze_thaw_model import FreezeThawGP
from robo.task.synthetic_functions.exp_decay import ExpDecay
from robo.maximizers.direct import Direct
from robo.acquisition.ei import EI
from robo.acquisition.information_gain_mc_freeze import InformationGainMC
from scipy.stats import norm

logger = logging.getLogger(__name__)


class FreezeThawBO(BaseSolver):

	def __init__(self, acquisition_func, freeze_thaw_model, maximize_func, task, initial_design=None, init_points=5, incumbent_estimation=None, basketOld_X=None, basketOld_Y=None, first_steps=3):
		"""
		Class for the Freeze Thaw Bayesian Optimization by Swersky et al.

		Parameters
		----------
		basketOld_X: ndarray(N,D)
		    Basket with the old configurations, with N as the number of configurations
		    and D as the number of dimensions
		basketOld_Y: ndarray(N,S)
		    Basket with the learning curves of the old configurations, with N as the number
		    of configurations and S as the number of steps
		first_steps: integer
		    Initial number of steps of the learning curves
		acquisition_func: BaseAcquisitionFunction Object
		    The acquisition function which will be maximized.
		freeze_thaw_model: ModelObject
		    Freeze thaw model that models our current
		    belief of the objective function.
		task: TaskObject
		    Task object that contains the objective function and additional
		    meta information such as the lower and upper bound of the search
		    space.
		maximize_func: MaximizerObject
		    Optimization method that is used to maximize the acquisition
		    function
		"""
		
		super(FreezeThawBO, self).__init__(acquisition_func, freeze_thaw_model, maximize_func, task)

		self.start_time = time.time()

		if initial_design == None:
			self.initial_design = init_random_uniform
		else:
			self.initial_design = initial_design

		self.X = None
		self.Y = None

		self.freezeModel = self.model

		self.task = task

		self.model_untrained = True

		if incumbent_estimation is None:
			self.estimator = BestObservation(self.model, self.task.X_lower, self.task.X_upper)
		else:
			self.estimator = incumbent_estimation

		self.incumbent = None
		self.incumbents = []
		self.incumbent_values = []
		self.init_points = init_points

		self.basketOld_X = basketOld_X
		self.basketOld_Y = basketOld_Y

		self.first_steps = first_steps

	def run(self, num_iterations=10, X=None, Y=None):
		"""
		Bayesian Optimization iterations

		Parameters
		----------
		num_iterations: int
		    How many times the BO loop has to be executed
		X: np.ndarray(N,D)
		    Initial points that are already evaluated
		Y: np.ndarray(N,1)
		    Function values of the already evaluated points

		Returns
		-------
		np.ndarray(1,D)
		    Incumbent
		np.ndarray(1,1)
		    (Estimated) function value of the incumbent
		"""

		init = init_random_uniform(self.task.X_lower, self.task.X_upper, self.init_points)
		self.basketOld_X = deepcopy(init)

		ys = np.zeros(self.init_points, dtype=object)
		for i in xrange(self.init_points):
			ys[i] = self.task.f(np.arange(1, 1 + self.first_steps), x=init[i, :])

		self.basketOld_Y = deepcopy(ys)
		# ys = y0, y1, y2, y3, y4
		# print type(ys)
		Y = np.zeros((len(ys), 1))
		for i in xrange(Y.shape[0]):
			Y[i, :] = ys[i][-1]
		print 'Y: ', Y
		#self.freezeModel = FreezeThawGP(x_train=self.basketOld_X, y_train=self.basketOld_Y)
		self.freezeModel.X = self.freezeModel.x_train = self.basketOld_X
		self.freezeModel.ys = self.freezeModel.y_train = self.basketOld_Y
		self.freezeModel.Y = Y
		self.freezeModel.actualize()
		task = ExpDecay()
		# ei = EI(freezeModel, X_lower, X_upper)
		# maximizer = Direct(ei, X_lower, X_upper)

		###############################From here onwards the iteration with for loop###########################################
		#######################################################################################################################

		for k in range(self.init_points, num_iterations):
			
			print '######################iteration nr: ', k, '#################################'

			res = self.choose_next(X=self.basketOld_X, Y=self.basketOld_Y, do_optimize=True)
			print 'res: ', res
			ig = InformationGainMC(model=self.freezeModel, X_lower=self.task.X_lower, X_upper=self.task.X_upper, sampling_acquisition=EI)
			ig.update(self.freezeModel, calc_repr=True)
			H = ig.compute()
			zb = deepcopy(ig.zb)
			lmb = deepcopy(ig.lmb)
			#H = 0
			print 'H: ', H
			# Fantasize over the old and the new configurations
			nr_old = self.init_points
			fant_old = np.zeros(nr_old)

			for i in xrange(nr_old):
				fv = self.freezeModel.predict(option='old', conf_nr=i)
				# print 'fv: ', fv
				fant_old[i] = fv[0]
				# print 'fant_old[i]: ', fant_old[i]
			
			nr_new = 1
			fant_new = np.zeros(nr_new)
			for j in xrange(nr_new):
				m, v = self.freezeModel.predict(xprime=res, option='new')
				fant_new[j] = m
				# print 'fant_new[j]: ', fant_new[j]

			Hfant = np.zeros(nr_old + nr_new)

			for i in xrange(nr_old):
				freezeModel = deepcopy(self.freezeModel)
				y_i = freezeModel.ys[i]
				y_i = np.append(y_i, np.array([fant_old[i]]), axis=0)
				freezeModel.ys[i] = freezeModel.y_train[i] = y_i
				freezeModel.Y[i, :] = y_i[-1]
				ig1 = InformationGainMC(model=freezeModel, X_lower=self.task.X_lower, X_upper=self.task.X_upper, sampling_acquisition=EI)
				ig1.actualize(zb, lmb)
				ig1.update(freezeModel)
				H1 = ig1.compute()
				Hfant[i] = H1

			freezeModel = deepcopy(self.freezeModel)
			print 'Hfant: ', Hfant
			freezeModel.X = np.append(freezeModel.X, res, axis=0)
			ysNew = np.zeros(len(freezeModel.ys) + 1, dtype=object)
			for i in xrange(len(freezeModel.ys)):
				ysNew[i] = freezeModel.ys[i]

			ysNew[-1] = np.array([fant_new[0]])
			freezeModel.ys = freezeModel.y_train = ysNew
			freezeModel.Y = np.append(freezeModel.Y, np.array([[fant_new[0]]]), axis=0)
			freezeModel.C_samples = np.zeros(
			            (freezeModel.C_samples.shape[0], freezeModel.C_samples.shape[1] + 1, freezeModel.C_samples.shape[2] + 1))
			freezeModel.mu_samples = np.zeros(
			    (freezeModel.mu_samples.shape[0], freezeModel.mu_samples.shape[1] + 1, 1))

			ig1 = InformationGainMC(model=freezeModel, X_lower=self.task.X_lower, X_upper=self.task.X_upper, sampling_acquisition=EI)
			ig1.actualize(zb, lmb)
			ig1.update(freezeModel)
			H1 = ig1.compute()
			Hfant[-1] = H1
			print 'Hfant: ', Hfant
			
			# Comparison of the different values
			infoGain = -(Hfant - H)
			winner = np.argmax(infoGain)
			print 'the winner is index: ', winner

			#run an old configuration and actualize basket
			#else run the new proposed configuration and actualize
			if winner != len(Hfant) - 1:
			    # run corresponding configuration for more one step
			    ytplus1 = self.task.f(t=len(self.basketOld_Y[winner]) + 1, x=self.basketOld_X[winner])
			    self.basketOld_Y[winner] = np.append(self.basketOld_Y[winner], ytplus1)
			else:
				ytplus1 = self.task.f(t=1, x=res[0])
				replace = get_min_ei(freezeModel, self.basketOld_X, self.basketOld_Y)
				self.basketOld_X[replace] = res[0]
				self.basketOld_Y[replace] = np.array([ytplus1])

			Y = getY(self.basketOld_Y)
			self.incumbent, self.incumbent_value = estimate_incumbent(Y, self.basketOld_X)
			self.incumbents.append(self.incumbent)
			self.incumbent_values.append(self.incumbent_values)
		return self.incumbent, self.incumbent_value

# which one to exclude from the old basket? The one with lowest ei
# print
# print 'basketOld_X: ', self.basketOld_X
# print
# print 'basketOld_Y: ', self.basketOld_Y
#"""
# Run winner for a certain amount of steps, let's say 1 step
# winner is an old configuration
# winner = 3 #for testing the substituion of an old config by a new one
# Hfant = np.arange(1,5) #for testing the substituion of an old config by a new one
# freezeModel = deepcopy(freezeModel) #for testing the substituion of an
# old config by a new one


	def choose_next(self, X=None, Y=None, do_optimize=True):
		"""
		Recommend a new configuration by maximizing the acquisition function

		Parameters
		----------
		num_iterations: int
		    Number of loops for choosing a new configuration
		X: np.ndarray(N,D)
		    Initial points that are already evaluated
		Y: np.ndarray(N,1)
		    Function values of the already evaluated points

		Returns
		-------
		np.ndarray(1,D)
		    Suggested point
		"""

		initial_design = init_random_uniform

		if X is None and Y is None:
			x = initial_design(self.task.X_lower, self.task.X_upper, N=1)

		elif X.shape[0] == 1:
			x = initial_design(self.task.X_lower, self.task.X_upper, N=1)
		else:
			try:
				self.freezeModel.train(X, Y, do_optimize=do_optimize)
			except:
				raise

			model_untrained = False

			self.acquisition_func.update(self.freezeModel)

			x = self.maximize_func.maximize()

		return x


def f(t, a=0.1, b=0.1, x=None):
	k=1e3
	if x is not None:
		a, b = x
	return k*a*np.exp(-b*t)

def getY(ys):
    Y = np.zeros((len(ys),1))
    for i in xrange(Y.shape[0]):
        Y[i,:] = ys[i][-1]
    return Y

def estimate_incumbent(Y, basketOld_X):
    best = np.argmin(Y)
    incumbent = basketOld_X[best]
    incumbent_value = Y[best]

    return incumbent[np.newaxis, :], incumbent_value[:, np.newaxis] 

def compute_ei(X, model, ys, basketOld_X, par=0.0):
    print 'in compute_ei X: ', X
    m, v = model.predict(X[None,:])
    # m = m[0]
    # v = v[0]
    print 'in compute_ei m: ', m, ' and v: ', v 

    Y = getY(ys)
    print 'in compute_ei Y: ', Y

    _, eta = estimate_incumbent(Y, basketOld_X)
    # eta = eta[0,0]
    print 'in compute_ei eta: ', eta

    s = np.sqrt(v)

    z = (eta - m - par) / s
    print 'in compute_ei z: ', z

    f = s * ( z * norm.cdf(z) +  norm.pdf(z))
    print 'in compute ei f: ', f
    return f

def get_min_ei(model, basketOld_X, basketOld_Y):
    nr = basketOld_X.shape[0]
    eiList = np.zeros(nr)
    for i in xrange(nr):
        val = compute_ei(basketOld_X[i], model, basketOld_Y, basketOld_X)
        # print'val in get_min_ei: ', val
        eiList[i] = val[0][0]
    minIndex = np.argmin(eiList)
    return minIndex
