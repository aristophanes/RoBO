# encoding=utf8
__author__ = "Tulio Paiva"
__email__ = "paivat@cs.uni-freiburg.de"

import numpy as np
from copy import deepcopy
import logging
import time
import os
import cPickle as pickle
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

"""
In this version 4 in comparision to the previous one 3, we are trying
to be the adequte to lasagne_logrg_task_freeze.py and simultaneously trying
to be more general for kinds of cases
"""

class FreezeThawBO(BaseSolver):

	def __init__(self,
		         acquisition_func,
		         freeze_thaw_model,
		         maximize_func,
		         task,
		         initial_design=None,
		         init_points=5,
		         incumbent_estimation=None,
		         basketOld_X=None,
		         basketOld_Y=None,
		         first_steps=3,
		         save_dir=None,
		         num_save=1,
		         train_intervall=1,
		         n_restarts=1,
		         pkl=False,
		         max_epochs=500, 
		         stop_epochs=False,
		         nr_epochs_inits=5,
		         nr_epochs_further=10):
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
		nr_epochs_init: integer 
		    Number of epochs for executing the task at the initial configurations
		nr_epochs_further: integer 
		    Number of epochs for executing the task at further configurations		
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
		self.incumbent_value = None
		self.incumbent_values = []
		self.init_points = init_points

		self.basketOld_X = basketOld_X
		self.basketOld_Y = basketOld_Y
		self.basket_files = list()
		self.basket_indices = list()
		self.directory = "temp_configs_" + self.task.base_name

		self.first_steps = first_steps
		self.nr_epochs_inits = nr_epochs_inits
		self.nr_epochs_further = nr_epochs_further

		self.time_func_eval = None
		self.time_overhead = None
		self.train_intervall = train_intervall

		self.num_save = num_save
		self.time_start = None
		self.pkl = pkl
		self.stop_epochs = stop_epochs
		self.max_epochs = max_epochs
		self.all_configs = dict()
		self.total_nr_confs = 0

		self.n_restarts = n_restarts
		self.runtime = []

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
		val_losses_all = []
		nr_epochs = 0

		self.create_file_paths()

		ys = np.zeros(self.init_points, dtype=object)
		##change: the task function should send the model file_path back and it should be stored here
		for i in xrange(self.init_points):
			#ys[i] = self.task.f(np.arange(1, 1 + self.first_steps), x=init[i, :]) ##change: that's not general
			#ys[i] = self.task.objective(nr_epochs=None, save_file_new=self.basket_files, save_file_old=None)
			self.task.set_save_modus(is_old=False, file_old=None, file_new=self.basket_files[i])
			self.set_epochs(self.nr_epochs_inits)
			#print 'init[i,:]: ', init[i,:]
			#_, ys[i] = self.task.objective_function(x=init[i,:][np.newaxis,:])
			conf_now = init[i,:]
			_, val_losses = self.task.evaluate(x=conf_now[np.newaxis,:])
			#storing configuration, learning curve, activity, index in the basketOld
			ys[i] = np.asarray(val_losses)
			self.all_configs[self.total_nr_confs] = [conf_now, ys[i], True, self.total_nr_confs]
			self.total_nr_confs+=1
			val_losses_all = val_losses_all + val_losses
			nr_epochs+=len(val_losses)
			#print 'in ftbo4 ys[i]: ', type(ys[i])
			

		self.basketOld_Y = deepcopy(ys)

		Y = np.zeros((len(ys), 1))
		for i in xrange(Y.shape[0]):
			Y[i, :] = ys[i][-1]

		self.freezeModel.X = self.freezeModel.x_train = self.basketOld_X
		self.freezeModel.ys = self.freezeModel.y_train = self.basketOld_Y
		self.freezeModel.Y = Y
		self.freezeModel.actualize()

		###############################From here onwards the iteration with for loop###########################################
		#######################################################################################################################

		for k in range(self.init_points, num_iterations):
			
			if self.stop_epochs and nr_epochs >= self.max_epochs:
				print 'Maximal number of epochs'
				break

			print '######################iteration nr: ', k, '#################################'

			#res = self.choose_next(X=self.basketOld_X, Y=self.basketOld_Y, do_optimize=True)
			#here just choose the next candidates, which will be stored in basketNew
			res = self.choose_next_ei(X=self.basketOld_X, Y=self.basketOld_Y, do_optimize=True)
			#res = res[0]
			print 'res: ', res
			ig = InformationGainMC(model=self.freezeModel, X_lower=self.task.X_lower, X_upper=self.task.X_upper, sampling_acquisition=EI)
			ig.update(self.freezeModel, calc_repr=True)
			H = ig.compute()
			zb = deepcopy(ig.zb)
			lmb = deepcopy(ig.lmb)
			print 'H: ', H
			# Fantasize over the old and the new configurations
			nr_old = self.init_points
			fant_old = np.zeros(nr_old)

			for i in xrange(nr_old):
				fv = self.freezeModel.predict(option='old', conf_nr=i)
				# print 'fv: ', fv
				fant_old[i] = fv[0]
				# print 'fant_old[i]: ', fant_old[i]
			
			nr_new = res.shape[0]

			fant_new = np.zeros(nr_new)
			for j in xrange(nr_new):
				m, v = self.freezeModel.predict(xprime=res[j,:], option='new')
				fant_new[j] = m

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

			print 'Hfant: ', Hfant
			print 'freezeModel.X: ', freezeModel.X
			print 'res: ', res

			for k in xrange(nr_new):

				freezeModel = deepcopy(self.freezeModel)

				freezeModel.X = np.append(freezeModel.X, res[k,:][np.newaxis,:], axis=0)
				ysNew = np.zeros(len(freezeModel.ys) + 1, dtype=object)
				for i in xrange(len(freezeModel.ys)): ##improve: do not use loop here, but some expansion
					ysNew[i] = freezeModel.ys[i]

				ysNew[-1] = np.array([fant_new[k]])
				freezeModel.ys = freezeModel.y_train = ysNew

				freezeModel.Y = np.append(freezeModel.Y, np.array([[fant_new[k]]]), axis=0)
				freezeModel.C_samples = np.zeros(
				            (freezeModel.C_samples.shape[0], freezeModel.C_samples.shape[1] + 1, freezeModel.C_samples.shape[2] + 1))
				freezeModel.mu_samples = np.zeros(
				    (freezeModel.mu_samples.shape[0], freezeModel.mu_samples.shape[1] + 1, 1))

				ig1 = InformationGainMC(model=freezeModel, X_lower=self.task.X_lower, X_upper=self.task.X_upper, sampling_acquisition=EI)
				ig1.actualize(zb, lmb)
				ig1.update(freezeModel)
				H1 = ig1.compute()
				Hfant[-(nr_new - k)] = H1 #?? why the initial - ?
				print 'Hfant: ', Hfant
			
			# Comparison of the different values
			infoGain = -(Hfant - H)
			winner = np.argmax(infoGain)
			print 'the winner is index: ', winner

			#run an old configuration and actualize basket
			if winner <= ((len(Hfant) - 1) - nr_new):
				print '###################### run old config ######################'
				# run corresponding configuration for more one step
				##change: the task function should send the model file_path back and it should be stored here
				#ytplus1 = self.task.f(t=len(self.basketOld_Y[winner]) + 1, x=self.basketOld_X[winner])
				#ytplus1 = self.task.f(t=len(self.basketOld_Y[winner]) + 1, x=self.basketOld_X[winner], save_file_old=self.basket_files[winner])
				self.task.set_save_modus(is_old=True, file_old=self.basket_files[winner], file_new=None)
				self.set_epochs(self.nr_epochs_further)

				if self.pkl:
					with open(self.basket_files[winner],'rb') as f:
						weights_final = pickle.load(f)
					W,b = weights_final
					self.task.set_weights(W=W, b=b)
				else:
					self.task.set_weights(self.basket_files[winner])
				#ytplus1 = self.task.objective_function(x=self.basketOld_X[winner])
				conf_to_run = self.basketOld_X[winner]
				_, val_losses= self.task.evaluate(x=conf_to_run[np.newaxis,:])
				ytplus1 = np.asarray(val_losses)
				val_losses_all = val_losses_all + val_losses
				nr_epochs+=len(val_losses)
				self.basketOld_Y[winner] = np.append(self.basketOld_Y[winner], ytplus1)
				
				index_now = self.basket_indices[winner]
				#self.all_configs[index_now] = [conf_to_run, self.basketOld_Y[winner], True, winner]
				self.all_configs[index_now][1] = self.basketOld_Y[winner]

			#else run the new proposed configuration and actualize
			else:
				print '###################### run new config ######################'
				winner = winner - nr_old
				##change: the task function should send the model file_path back and it should be saved here
				#ytplus1 = self.task.f(t=1, x=res[winner])
				if self.pkl:
					file_path = "config_" + str(len(self.basket_files)) + ".pkl"
				else:
					file_path = "config_" + str(len(self.basket_files))
				file_path = os.path.join(self.directory, file_path)
				#ytplus1 = self.task.f(t=1, x=res[winner], save_file_new=file_path)
				self.task.set_save_modus(is_old=False, file_old=None, file_new=file_path)
				self.set_epochs(self.nr_epochs_further)
				#ytplus1 = self.task.objective_function(x=res[winner])
				_, val_losses = self.task.evaluate(x=res[winner][np.newaxis,:])
				ytplus1 = np.asarray(val_losses)
				val_losses_all = val_losses_all + val_losses
				nr_epochs+=len(val_losses)
				replace = get_min_ei(freezeModel, self.basketOld_X, self.basketOld_Y)
				self.basketOld_X[replace] = res[winner]
				if type(ytplus1) != np.ndarray:
					self.basketOld_Y[replace] = np.array([ytplus1])
				else:
					self.basketOld_Y[replace] = ytplus1
				self.basket_files[replace] = file_path

				#deactivate the configuration which is being replaced
				self.all_configs[self.basket_indices[replace]][2] = False
				self.all_configs[self.basket_indices[replace]][3] = -1
				#actualize the indices table
				self.total_nr_confs+=1
				self.basket_indices[replace]=self.total_nr_confs
				#add new configuration with learning curve and index
				self.all_configs[self.total_nr_confs] = [conf_to_run, self.basketOld_Y[winner], True, replace]

			Y = getY(self.basketOld_Y)
			self.incumbent, self.incumbent_value = estimate_incumbent(Y, self.basketOld_X)
			self.incumbents.append(self.incumbent)
			self.incumbent_values.append(self.incumbent_value)

			with open('val_losses_all.pkl', 'wb') as f:
				pickle.dump(val_losses_all, f)

			with open('all_configs.pkl','wb') as c:
				pickle.dump(self.all_configs, c)

		return self.incumbent, self.incumbent_value


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


	def choose_next_ei(self, X=None, Y=None, do_optimize=True, N=10, M=3):
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
		N: integer
		    Number of configuration samples
		M: integer
		    Chosen configurations
		Returns
		-------
		np.ndarray(1,D)
		    Suggested point
		"""

		initial_design = init_random_uniform

		self.freezeModel.train(X, Y, do_optimize=do_optimize)

		x = initial_design(self.task.X_lower, self.task.X_upper, N=N)
		#print 'in choose_next_ei x: ', x.shape

		ei_list = np.zeros(x.shape[0])

		for i in xrange(N):
			ei_list[i] = compute_ei(X=x[i,:], model=self.freezeModel, ys=Y, basketOld_X=X)

		#print 'in choose_next_ei ei_list: ', ei_list.shape

		sort = np.argsort(ei_list)

		highest_confs = x[sort][-M:]
		#print 'in choose_next_ei highest_confs: ', highest_confs.shape

		return highest_confs

	def create_file_paths(self):
		if not os.path.exists(self.directory):
			os.makedirs(self.directory)

		for index in xrange(self.init_points):
			if self.pkl:
				file_path = "config_" + str(index) + ".pkl"
			else:
				file_path = "config_" + str(index)
			self.basket_files.append(os.path.join(self.directory, file_path))
			self.basket_indices.append(index)#that's exactly true here, thereafter there are changes
		#print self.basket_files


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
    #print 'in compute_ei X: ', X
    m, v = model.predict(X[None,:])
    # m = m[0]
    # v = v[0]
    #print 'in compute_ei m: ', m, ' and v: ', v 

    Y = getY(ys)
    #print 'in compute_ei Y: ', Y

    _, eta = estimate_incumbent(Y, basketOld_X)
    # eta = eta[0,0]
    #print 'in compute_ei eta: ', eta

    s = np.sqrt(v)

    z = (eta - m - par) / s
    #print 'in compute_ei z: ', z

    f = s * ( z * norm.cdf(z) +  norm.pdf(z))
    #print 'in compute ei f: ', f
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
