# encoding=utf8
__author__ = "Tulio Paiva"
__email__ = "paivat@cs.uni-freiburg.de"

import scipy
import numpy as np
import logging
from scipy import optimize
from robo.models.base_model import BaseModel
from sklearn.metrics import mean_squared_error as mse
import time
import matplotlib.pyplot as pl
from numpy.linalg import inv
import emcee
import scipy.stats as sps
from scipy.optimize import minimize
from numpy.linalg import solve
from math import exp
from scipy.linalg import block_diag
from robo.priors.base_prior import BasePrior, TophatPrior, LognormalPrior, HorseshoePrior

logger = logging.getLogger(__name__)

class FreezeThawGP(BaseModel):
	
	def __init__(self,
	             x_train=None,
	             y_train=None,
	             x_test=None,
	             y_test=None,
	             sampleSet=None):
		
		"""
        Interface to the freeze-thawn GP library. The GP hyperparameter are obtained
        by integrating out the marginal loglikelihood over the GP hyperparameters.

        Parameters
        ----------
	    x_train: ndarray(N,D)
			The input training data for all GPs
		y_train: ndarray(T,N)
			The target training data for all GPs. The ndarray can be of dtype=object,
			if the curves have different lengths
		x_test: ndarray(*,D)
			The current test data for the GPs, where * is the number of test points
        sampleSet : ndarray(S,H)
			Set of all GP hyperparameter samples (S, H), with S as the number of samples and H the number of
			GP hyperparameters. This option should only be used, if the GP-hyperparameter samples already exist
		"""

		self.X = x_train
		self.y = y_train
		self.x_test = x_test
		self.y_test = y_test
		
		if sampleSet != None:
			self.ps = PredSamples(samples=sampleSet, x_train = self.X, y_train = self.y, x_test = self.X, predHyper=True, predOld=True, predNew=True, invChol=True, samenoise=True)
	
	def train(self, X=None, Y=None, do_optimize=True):
		"""
        Estimates the GP hyperparameter by integrating out the marginal
        loglikelihood over the GP hyperparameters	
        
        Parameters
        ----------
	    x_train: ndarray(N,D)
			The input training data for all GPs
		y_train: ndarray(T,N)
			The target training data for all GPs. The ndarray can be of dtype=object,
			if the curves have different lengths.	
        """
        
		if X != None:
			self.X = X
		if Y != None:
			self.Y = Y
		
		if do_optimize:	
			lik = LikIntegrate(y_train=self.y, invChol = True, horse=True, samenoise=True)
			sampleSet = lik.create_configs(x_train=self.X, y_train=self.y, hyper_configs=12, chain_length=100, burnin_steps=100)
			
			self.ps = PredSamples(samples=sampleSet, x_train = self.X, y_train = self.y, x_test = self.X, predHyper=True, predOld=True, predNew=True, invChol=True, samenoise=True)
		
	
	def predict(self, X):
		"""
		Predicts asymptotic mean and std2 for configurations X. The prediction is averaged for
		all GP hyperparameter samples. They are integrated out.
		
		Parameters
		----------
		X: ndarray(*, D)
				the configurations for which the mean and the std2 are being predicted
		
		Returns
		-------
		mean: ndarray(*,1)
			  predicted mean for every configuration in X
		std2: ndarray(*,1)
			  predicted std2 for every configuration in X
		"""
		mean, std2, _ = self.ps.pred_hyper(X)
		return mean, std2

	def predictive_new(self, configuration, steps=1):
		"""
		Predicts mean and std2 for configurations completely new configuration X. The prediction is averaged for
		all GP hyperparameter samples. They are integrated out.
		
		Parameters
		----------
		configuration: ndarray(*, D)
				the new configurations for which the mean and the std2 are being predicted
		steps: integer
				the number of steps to be predicted, from the first step onwards
		Returns
		-------
		mean: ndarray(*,1)
			  predicted mean for every configuration in configuration
		std2: ndarray(*,1)
			  predicted std2 for every configuration in configuration
		"""
		
		mean, std2 = self.ps.pred_new(steps=steps, xprime=configuration, asy=True)
		
		return mean, std2

	def predictive_old(self, conf_nr, steps, fro=None):
		"""
		Predicts mean and std2 for an already active configuration of number conf_nr. The prediction is averaged for
		all GP hyperparameter samples. They are integrated out. If fro is None, then the prediction is done from the 
		last step of the curve for a certain number of further steps. Otherwise fro should be set to a value from where
		to begin the prediction.
		
		Parameters
		----------
		conf_nr: integer
				the number of the configuration for which the mean and the std2 are being predicted
		steps: integer
				the number of steps to be predicted
		fro: integer
				initial step from which to predict further steps of an already initialized configuration
		
		Returns
		-------
		meansCur: ndarray(*,1)
			  predicted means for all predicted steps of configuration number conf_nr
		std2sCur: ndarray(*,1)
			  predicted std2 for all predicted steps of configuration number conf_nr
		"""
		if fro == None:
			meansCur, std2sCur, divbyCur = self.ps.pred_old(conf_nr=conf_nr + 1, steps=steps)
		else:
			meansCur, std2sCur, divbyCur = self.ps.pred_old(conf_nr=conf_nr + 1, steps=steps, fro=fro)
		
		return meansCur, std2sCur

class PredSamples(object):
	"""
	A class for controlling all different types of GPs, and doing predictions by integrating out
	the different samples of the GP hyperparameters
	
	Parameters
	----------
	samples: ndarray(number_samples, D)
			 All samples generated by the MCMC from LikIntegrate. Each sample contain samples 
			 for every GP hyperparameter
	x_train: ndarray(N, D)
			 The input training data ofor all GPs
	y_train: ndarray(T, N)
			 The target training data for all GPs
	x_test: ndarray(*, D)
			 The current test data of the GPs
	predHyper: boolean
			   Whether the posterior predictive over the hyperparameter is to be activated
	predOld: boolean
			   Whether the posterior predictive over the training curve of an old configuration is to be activated
	predNew: boolean
			   Whether the posterior predictive over the training curve of a new configuration is to be activated
	"""	
	def __init__(self,
				 samples,
				 x_train = None,
				 y_train = None,
				 x_test = None,
				 predHyper=True,
				 predOld=True,
				 predNew=True,
				 invChol=True,
				 samenoise=True):
		
		self.samples = samples
		self.x_train = x_train
		self.y_train = y_train
		self.x_test = x_test
		self.invChol = invChol
		self.samenoise = samenoise
		
		if predHyper:
			self.ph = PredictiveHyper(x_train, y_train, x_test, invChol=self.invChol, samenoise=self.samenoise)
		else:
			self.ph= None
		
		if predOld:
			self.po = PredictiveOld(x_train, y_train, x_test, samenoise=self.samenoise)
		else:
			self.po = None
		
		if predNew:
			self.pn = PredictiveNew(x_train, y_train, x_test, samenoise=self.samenoise)
		else:
			self.pn = None
		
		self.C_samples = np.zeros((samples.shape[0], self.x_train.shape[0], self.x_train.shape[0]))
		self.mu_samples = np.zeros((samples.shape[0], self.x_train.shape[0], 1))
		self.activated = False
	
	
	def pred_hyper(self, xprime=None):
		"""
		Predicts mean and std2 for new configurations xprime. The prediction is averaged for
		all GP hyperparameter samples. They are integrated out.
		
		Parameters
		----------
		xprime: ndarray(*, D)
				the new configurations for which the mean and the std2 are being predicted
		
		Returns
		-------
		mean: ndarray(*,1)
			  predicted mean for every new configuration in xprime
		std2: ndarray(*,1)
			  predicted std2 for every new configuration in xprime
		divby: integer
			   number of GP hyperparameter samples which deliver acceptable results for mean
			   and std2. 
		"""
		if xprime != None:
			self.x_test = xprime
		
		samples_val = []
		C_valid = []
		mu_val = []
		means_val = []
		std2s_val = []
		
		divby = self.samples.shape[0]
		
		
		for i in xrange(self.samples.shape[0]):
			self.ph.setGpHypers(self.samples[i])
			pre = self.ph.predict_asy(xprime)
			#print 'pre: ', pre
			if pre!=None:
				mean_one, std2_one, C, mu = pre
				#print 'mean_one: ', mean_one
				means_val.append(mean_one.flatten())
				#print 'means_val: ', means_val
				std2s_val.append(std2_one.flatten())
				C_valid.append(C)
				mu_val.append(mu)
				samples_val.append(self.samples[i])
			else:
				divby -= 1 
				#print 'bad: ', divby
		
		mean_temp = np.zeros((divby, xprime.shape[0]))
		std2_temp = np.zeros((divby, xprime.shape[0]))
		
		if(divby < self.samples.shape[0]):
			self.C_samples = np.zeros((divby, self.C_samples.shape[1], self.C_samples.shape[2]))
			self.mu_samples = np.zeros((divby, self.mu_samples.shape[1], self.mu_samples.shape[2]))
			self.samples = np.zeros((divby, self.samples.shape[1]))
		
		
		for j in xrange(divby):
			mean_temp[j,:] = means_val[j]
			std2_temp[j,:] = std2s_val[j]
			self.C_samples[j, ::] = C_valid[j]
			self.mu_samples[j, ::] = mu_val[j]
			self.samples[j, ::] = samples_val[j]
		

		

		mean = np.mean(mean_temp, axis=0)
		std2 = np.mean(std2_temp, axis=0) + np.mean(mean_temp**2, axis=0)
		std2 -= mean**2
		
		self.activated = True
		self.asy_mean = mean
		
		return mean, std2, divby

		
	
	def pred_old(self, conf_nr, steps, fro=None):
		"""
		Here conf_nr is from 1 onwards. That's why we are using mu_n = mu[conf_nr - 1, 0] in the for-loop
		"""
		
		if self.activated:
			
			means_val = []
			std2s_val = []
			divby = self.samples.shape[0]
			
			yn = self.y_train[conf_nr -1]
			if fro == None:
				t = np.arange(1, yn.shape[0] +1)
				tprime = np.arange(yn.shape[0] +1, yn.shape[0] +1 +steps)
			else:
				t = np.arange(1, fro)
				tprime = np.arange(fro, fro + steps)				
			
	
			for i in xrange(self.samples.shape[0]):
				#print 'samples: ', self.samples
				self.po.setGpHypers(self.samples[i])
				
				mu = self.mu_samples[i, ::]
				mu_n = mu[conf_nr - 1, 0]
				
				C = self.C_samples[i, ::]
				Cnn = C[conf_nr - 1, conf_nr -1]
				
				
				pre = self.po.predict_new_point1(t, tprime, yn, mu_n, Cnn)

				if pre!= None:
					mean_one, std2_one = pre
					means_val.append(mean_one.flatten())
					std2s_val.append(std2_one.flatten())
				else:
					divby-=1
			
			mean_temp = np.zeros((divby, steps))
			std2_temp = np.zeros((divby, steps))
			
			for j in xrange(divby):
				mean_temp[j,:] = means_val[j]
				std2_temp[j,:] = std2s_val[j]

			mean = np.mean(mean_temp, axis=0)
			std2 = np.mean(std2_temp, axis=0) + np.mean(mean_temp**2, axis=0)
			std2 -= mean**2
	
			
			return mean, std2, divby
		
		else:
			raise Exception
		
	
	def pred_new(self, steps=13, xprime=None, y=None, asy=False):
		"""
		Params
		------
		asy: Whether the asymptotic has already been calculated or not.
		"""
		if xprime != None:
			self.x_test = xprime
		
		#Not redundant here. The  PredictiveHyper object is already created. In case kx has already been calculate
		#it's not going to be calculated a second time. 
		if asy==False:
			asy_mean, std2star, _ = self.pred_hyper2(xprime)
		else:
			asy_mean = self.asy_mean
		
		mean_temp = np.zeros((self.samples.shape[0], steps))
		std2_temp = np.zeros((self.samples.shape[0], steps))
	
		for i in xrange(self.samples.shape[0]):
			self.pn.setGpHypers(self.samples[i])
				
			mean_one, std2_one = self.pn.predict_new_point2(steps, asy_mean[1], y)
			mean_temp[i, :] = mean_one.flatten()
			std2_temp[i, :] = std2_one.flatten()
			
		mean = np.mean(mean_temp, axis=0)
		std2 = np.mean(std2_temp, axis=0) + np.mean(mean_temp**2, axis=0)
		std2 -= mean**2
		return mean, std2

class LikIntegrate(object):
	'''
	Sampling of GP hyperparameter samples from the log likelihood through the GP MCMC
	
	Parameters
	----------
	y_train: ndarray(N, dtype=object)
		All training curves all together. Each training curve can have a different number of steps.
	invChol: boolean
		Use the cholesky decomposition for calculating the covariance matrix inversion
	horse: boolean
		Use the horseshoe prior for sampling the noise values
	samenoise: boolean
		Assume noise of GPs over learning curve and over configurations are the same
	'''
	
	def __init__(self,
				 y_train,
				 invChol = True,
				 horse=True,
				 samenoise=True):
		
		self.y_train = y_train
		self.invChol = invChol
		self.horse = horse
		self.samenoise = samenoise
	
	def samples_norm(self, n_samples):
		"""
		Samples from the lognorm distribution
		Parameters
		----------
		n_samples: scalar | tuple
			The shape of the samples from the lognormal distribution
		Returns
		-------
		ndarray(n_samples)
			The samples from the lognorm
		"""
		return np.random.lognormal(mean=0.,
							   sigma=1,
							   size=n_samples)
	
	def samples_noise(self, n_samples):
		"""
		Samples noise from the lognorm distribution for the kernel_hyper and kernel_curve
		Parameters
		----------
		n_samples: scalar | tuple
			The shape of the samples from the lognormal distribution
		Returns
		-------
		ndarray(n_samples)
			The noise samples from the lognorm
		"""
		return np.random.lognormal(mean=0.,
							   sigma=1,
							   size=n_samples)
	
	def samples_horse(self, n_samples, scale=0.1, rng=None):
		if rng is None:
			rng = np.random.RandomState(42)
		
		lamda = np.abs(rng.standard_cauchy(size=n_samples))
		#p0 = np.log(np.abs(rng.randn() * lamda * scale))
		p0 = np.abs(np.log(np.abs(rng.randn() * lamda * scale)))
		return p0
	
	
	def samples_uniform(self, n_samples):
		"""
		Samples values between 0 and 10 from the uniform distribution
		Parameters
		----------
		n_samples: scalar | tuple
			The shape of the samples from the uniform distribution
		Returns
		-------
		ndarray(n_samples)
			The samples from the uniform distribution
		"""
		return np.log(np.random.uniform(0, 10, n_samples))
	
	def sampleMconst(self, n_samples=(1,1)):
		"""
		Samples values between y_min and y_max from the uniform distribution
		Parameters
		----------
		n_samples: scalar | tuple
			The shape of the samples from the uniform distribution
		Returns
		-------
		ndarray(n_samples)
			The samples from the uniform distribution between y_min and y_max
		"""
		y = self.getYvector()
		return np.log(np.random.uniform(np.min(y), np.max(y), n_samples))
		

	def getYvector(self):
		"""
		Transform the y_train from type ndarray(N, dtype=object) to ndarray(T, 1).
		That's necessary for doing matrices operations
		
		Returns
		-------
		y_vec: ndarray(T,1)
			An array containing all loss measurements of all training curves. They need
			to be stacked in the same order as in the configurations array x_train
		"""
		y_vec = np.array([self.y_train[0]])
		for i in xrange(1, self.y_train.shape[0]):
			y_vec = np.append(y_vec, self.y_train[i])
		return y_vec.reshape(-1, 1)


	def create_configs(self, x_train, y_train, hyper_configs=40, chain_length=200, burnin_steps=200):
		"""
		MCMC sampling of the GP hyperparameters
		
		Parameters
		----------
		x_train: ndarray(N, D)
			The input training data.
		y_train: ndarray(N, dtype=object)
			All training curves. Their number of steps can diverge
		hyper_configs: integer
			The number of walkers
		chain_length: integer
			The number of chain steps 
		burnin_steps: integer
			The number of MCMC burning steps
		
		Results
		-------
		samples: ndarray(hyper_configs, number_gp_hypers)
			The desired number of samples for all GP hyperparameters
		"""
		
		#number of length scales
		flex = x_train.shape[-1]

		if not self.samenoise:
			#theta0, noiseHyper, noiseCurve, alpha, beta, m_const
			fix = 5 
		else:
			#theta0, noise, alpha, beta, m_const
			fix = 4
		
		pdl = PredLik(x_train, y_train, invChol=self.invChol, horse=self.horse, samenoise=self.samenoise)
		
		samples = np.zeros((hyper_configs, fix+flex))
		
		sampler = emcee.EnsembleSampler(hyper_configs, fix+flex, pdl.marginal_likelihood)
		
		#sample length scales for GP over configs
		p0a = self.samples_uniform((hyper_configs, flex))
		
		#sample amplitude for GP over configs and alpha e beta for GP over curve
		lnPrior = LognormalPrior(sigma=0.1, mean=0.0)
		
		p0b = self.samples_norm((hyper_configs, 3))
		print 'p0b: ', p0b
		p0b = lnPrior.sample_from_prior(n_samples=(hyper_configs, 3))
		print 'p0b: ', p0b
		
		p0 = np.append(p0a, p0b, axis=1)
		
		if not self.samenoise:
			if not self.horse:
				p0d = self.samples_noise((hyper_configs, 2))
			else:
				p0d = self.samples_horse((hyper_configs, 2))
		else:
			if not self.horse:
				p0d = self.samples_noise((hyper_configs, 1))
			else:
				p0d = self.samples_horse((hyper_configs, 1))
		
		
		p0 = np.append(p0, p0d, axis=1)
		
		p0, _, _ = sampler.run_mcmc(p0, burnin_steps)
		
		
		pos, prob, state = sampler.run_mcmc(p0, chain_length)
		
		p0 = pos
		
		samples = sampler.chain[:, -1]
		
		return np.exp(samples)

class PredLik(object):
	"""
	Class for the marginal likelihood of the GPs for the whole BO framework.
	
	Parameters
	----------
	x_train: ndarray(N,D)
		The input training data for all GPs
	y_train: ndarray(T,N)
		The target training data for all GPs
	x_test: ndarray(*,D)
		The current test data for the GPs
	theta_d = ndarray(D)
		Hyperparameters of the GP over hyperparameter configurations
	theta0: float
		Hyperparameter of the GP over hyperparameter configurations
	alpha: float
		Hyperparameter of the GP over training curves
	beta: float
		Hyperparameter of the GP over training curves

	"""
	def __init__(self,
	             x_train=None,
	             y_train=None,
	             x_test=None,
	             invChol=True,
	             horse=True,
	             samenoise=False):
		
		self.x_train = x_train
		self.y_train = y_train
		self.y = y_train
		self.x_test = x_test
		self.tetha_d = None
		self.theta0 = None
		self.alpha = None
		self.beta = None
		self.invChol = invChol
		self.horse = horse
		self.samenoise = samenoise
		self.m_const = None
		
	def inverse(self, chol):
		''' 
		Once one already has the cholesky of K, one can use this function for calculating the inverse of K
		
		:param chol: the cholesky decomposition of K
		:return: the inverse of K
		'''
		
		inve = 0
		error_k = 1e-25
		while(True):
			try:
				choly = chol + error_k*np.eye(chol.shape[0])
				inve = solve(choly.T, solve(choly, np.eye(choly.shape[0])))
				break
			except np.linalg.LinAlgError:
				error_k*=10
		return inve
	
	def invers(self, K):
		if self.invChol:
			invers = self.inverse_chol(K)
		else:
			try:
				invers = np.linalg.inv(K)
			except:
				invers=None
		
		return invers
	
	def inverse_chol(self, K):
		''' 
		Once one already has the cholesky of K, one can use this function for calculating the inverse of K
		
		:param chol: the cholesky decomposition of K
		:return: the inverse of K
		'''
		
		chol = self.calc_chol(K)
		if chol==None:
			return None
		
		inve = 0
		error_k = 1e-25
		while(True):
			try:
				choly = chol + error_k*np.eye(chol.shape[0])
				inve = solve(choly.T, solve(choly, np.eye(choly.shape[0])))
				break
			except np.linalg.LinAlgError:
				error_k*=10
		return inve

	def inversing(self, chol):
		inve = 0
		error_k = 1e-25
		once = False
		while(True):
			try:
				if once == True:
					choly = chol + error_k*np.eye(chol.shape[0])
				else:
					choly = chol
					once = True
				
				inve = solve(choly.T, solve(choly, np.eye(choly.shape[0])))
				break
			except np.linalg.LinAlgError:
				error_k*=10
		return inve
	
	def calc_chol(self, K):
		"""
		Calculates the cholesky decomposition of the positive-definite matrix K
		
		Parameters
		----------
		K: ndarray
		   Its dimensions depend on the inputs x for the inputs. len(K.shape)==2
		
		Returns
		-------
		chol: ndarray(K.shape[0], K.shape[1])
			  The cholesky decomposition of K
		"""
		
		error_k = 1e-25
		chol = None
		index = 0
		found = True
		while(index < 100):
			try:
				Ky = K + error_k*np.eye(K.shape[0])
				chol = np.linalg.cholesky(Ky)
				found = True
				break
			except np.linalg.LinAlgError:
				error_k*=10
				found = False
			index+=1
			
		if found:
			return chol
		else:
			return None

	def calc_chol3(self, K):
		"""
		Calculates the cholesky decomposition of the positive-definite matrix K
		
		Parameters
		----------
		K: ndarray
		   Its dimensions depend on the inputs x for the inputs. len(K.shape)==2
		
		Returns
		-------
		chol: ndarray(K.shape[0], K.shape[1])
			  The cholesky decomposition of K
		"""
		#print 'K: ', K
		error_k = 1e-25
		chol = None
		once = False
		index = 0
		found = True
		while(index < 100):
			try:
				#print 'chol index: ', index
				if once == True:
					#print 'once == True'
					Ky = K + error_k*np.eye(K.shape[0])
				else:
					#print 'once == False'
					#Ky = K
					Ky = K + error_k*np.eye(K.shape[0])
					once = True
				chol = np.linalg.cholesky(Ky)
				#print 'chol: ', chol
				found = True
				break
			except np.linalg.LinAlgError:
				#print 'except'
				error_k*=10
				found = False
			#print 'index: ', index
			index+=1
		if found:
			#print 'it is found'
			return chol
		else:
			#print 'not found'
			return None
	

	
	def get_mconst(self):
		m_const = np.zeros((len(self.y_train), 1))
		for i in xrange(self.y_train.shape[0]):
			mean_i = np.mean(self.y_train[i], axis=0)
			m_const[i,:] = mean_i
		
		return m_const
	
	def kernel_curve(self, t, tprime, alpha, beta):
		"""
		Calculates the kernel for the GP over training curves
		
		Parameters
		----------
		t: ndarray
			learning curve steps
		tprime: ndarray
			learning curve steps. They could be the same or different than t, depending on which covariace is being built
		
		Returns
		-------
		ndarray
			The covariance of t and tprime
		"""

		try:
			result = np.power(self.beta, self.alpha)/np.power(((t[:,np.newaxis] + tprime) + self.beta), self.alpha)
			#print 'result1 in kernel_curve: ', result
			#result = result + np.eye(result.shape)*self.noiseCurve
			result = result + np.eye(M=result.shape[0], N=result.shape[1])*self.noiseCurve
			#print 'result2 in kernel_curve: ', result
			return result
		except:
			return None
		
	def kernel_hyper(self, x, xprime, theta_d, theta0):
		"""
		Calculates the kernel for the GP over configuration hyperparameters
		
		Parameters
		----------
		x: ndarray
			Configurations of hyperparameters, each one of shape D
		xprime: ndarray
			Configurations of hyperparameters. They could be the same or different than x, 
			depending on which covariace is being built
		
		Returns
		-------
		ndarray
			The covariance of x and xprime
		"""
		#print 'x.shape: ', x.shape
		#print 'xprime.shape: ', xprime.shape
		#print 'theta_d.shape: ', theta_d.shape
		#print 'theta0: ', theta0
		try:
			r2 = np.sum(((x[:, np.newaxis] - xprime)**2)/self.theta_d**2, axis=-1)
			#print 'r2: ', r2
			fiveR2 = 5*r2
			result = self.theta0*(1 + np.sqrt(fiveR2) + (5/3.)*fiveR2)*np.exp(-np.sqrt(fiveR2))
			#print 'result1: ', result
			result = result + np.eye(M=result.shape[0], N=result.shape[1])*self.noiseHyper
			#print 'result2: ', result
			return result
		except:
			return None

	def getKt(self, y):
		"""
		Caculates the blockdiagonal covariance matrix Kt. Each element of the diagonal corresponds
		to a covariance matrix Ktn
		
		Parameters
		----------
		y: ndarray(N, dtype=object)
		   All training curves stacked together
		
		Returns
		-------

		"""
		ktn = self.getKtn(y[0])
		O = block_diag(ktn)
		
		for i in xrange(1, y.shape[0]):
			#print 'in getKt() y[i]: ', y[i]
			ktn = self.getKtn(y[i])
			#print 'in getKt() ktn: ', ktn
			#print
			O = block_diag(O, ktn)
		return O
	
	def getKtn(self, yn):
		t = np.arange(1, yn.shape[0]+1)
		#not yet using the optimized parameters here
		#print 't range: ', t
		ktn = self.kernel_curve(t, t, 1., 1.)
		#It's already returning None when necessary
		return ktn

	def calc_Lambda(self, y):
		'''
		Calculates Lambda according to the following: Lamda = transpose(O)*inverse(Kt)*O
		= diag(l1, l2,..., ln)=, where ln = transpose(1n)*inverse(Ktn)*1n
		
		Parameters
		----------
		y: ndarray(T,1)
			Vector with all training curves stacked together, in the same order as in the configurations array x_train
		
		Returns
		-------
		Lambda: ndarray(N, N)
				Lamda is used in several calculations in the BO framework
		'''
		dim = y.shape[0]
		Lambda = np.zeros((dim, dim))
		index = 0
		for yn in y:
			t = np.arange(1, yn.shape[0]+1)
			
			ktn = self.kernel_curve(t, t, 1.0, 1.0)
			if ktn == None:
				return None
				
			ktn_inv = self.invers(ktn)
			if ktn_inv==None:
				return None
			one_n = np.ones((ktn.shape[0], 1))
			Lambda[index, index] = np.dot(one_n.T, np.dot(ktn_inv, one_n))
			
			index+=1
		
		return Lambda
		
 	
	
	def lambdaGamma(self, y, m_const):
		dim = y.shape[0]
		Lambda = np.zeros((dim, dim))
		gamma = np.zeros((dim, 1))
		for i, yn in enumerate(y):
			t = np.arange(1, yn.shape[0]+1)
			ktn = self.kernel_curve(t, t, 1., 1.)
			if ktn == None:
				return None, None
			ktn_inv = self.invers(ktn)
			if ktn_inv == None:
				return None, None
			one_n = np.ones((ktn.shape[0], 1))
			Lambda[i, i] = np.dot(one_n.T, np.dot(ktn_inv, one_n))
			gamma[i, 0] = np.dot(one_n.T, np.dot(ktn_inv, yn - m_const[i]))
		
		return Lambda, gamma

	def lambdaGamma2(self, m_const):
		"""
		Difference here is that the cholesky decomposition is calculated just once for the whole Kt and thereafter
		we solve the linear system for each Ktn.
		"""
		Kt = self.getKt(self.y)
		#print 'Kt.shape: ', Kt.shape
		self.Kt_chol = self.calc_chol3(Kt)
		if self.Kt_chol == None:
			return None, None
		dim = self.y.shape[0]
		Lambda = np.zeros((dim, dim))
		gamma = np.zeros((dim, 1))
		index = 0
		for i, yn in enumerate(self.y):
			lent = yn.shape[0]
			ktn_chol = self.Kt_chol[index:index+lent, index:index+lent]
			#print 'ktn_chol.shape: ', ktn_chol.shape
			index+=lent
			ktn_inv = self.inversing(ktn_chol)
			if ktn_inv == None:
				return None, None
			one_n = np.ones((ktn_inv.shape[0], 1))
			#print 'one_n.shape: ', one_n.shape
			Lambda[i, i] = np.dot(one_n.T, np.dot(ktn_inv, one_n))
			gamma[i, 0] = np.dot(one_n.T, np.dot(ktn_inv, yn - m_const[i]))
		
		return Lambda, gamma
		
	def calc_gamma(self, y, m_const):
		'''
        Calculates gamma according to the following: gamma = transpose(O)*inverse(Kt)*(y - Om),
		where each gamma element gamma_n = transpose(1n)*inverse(Ktn)*(y_n -m_n)
		
		Parameters
		----------
		y: ndarray(T,1)
			Vector with all training curves stacked together, in the same order as in the configurations array x_train
		m_const: float
			the infered mean of f, used in the joint distribution of f and y.
		
		Returns
		-------
		gamma: ndarray(N, 1)
			gamma is used in several calculations in the BO framework
		'''
		dim = y.shape[0]
		gamma = np.zeros((dim, 1))
		index = 0
		for i, yn in enumerate(y):
			t = np.arange(1, yn.shape[0]+1)
			
			ktn = self.kernel_curve(t, t, 1.0, 1.0)
			if ktn == None:
				return None
	
			ktn_inv = self.invers(ktn)
			if ktn_inv == None:
				return None
			one_n = np.ones((ktn.shape[0], 1))
			gamma[index, 0] = np.dot(one_n.T, np.dot(ktn_inv, yn - m_const[i]))

			index+=1
	
		
		return gamma 
	
	def predict_asy(self, x, xprime, y):
		'''
		Given new configuration xprime, it predicts the probability distribution of
		the new asymptotic mean, with mean and covariance of the distribution
		
		Parameters
		----------
		xprime: ndarray(number_configurations, D)
			The new configurations, of which the mean an the std2 are being predicted
		
		Returns
		-------
		mean: ndarray(len(xprime))
			predicted means for each one of the test configurations
		std2: ndarray(len(xprime))
			predicted std2s for each one of the test configurations
		C: ndarray(N,N)
			The covariance of the posterior distribution. It is used several times in the BO framework
		mu: ndarray(N,1)
			The mean of the posterior distribution. It is used several times in the BO framework
		'''
		theta_d = np.ones(x.shape[-1])
		kx_star = self.kernel_hyper(x, xprime, theta_d, 1.0)
		kx = self.kernel_hyper(x, x, theta_d, 1.0)

		m_xstar = xprime.mean(axis=1).reshape(-1, 1)
		m_xstar = np.zeros(m_xstar.shape)
		m_const = self.get_mconst(y)
		m_const = np.zeros(m_const.shape)
		cholx = self.calc_chol(kx)
		kx_inv = self.inverse(cholx)
		Lambda = self.calc_Lambda(y)
		C_inv = kx_inv + Lambda
		C_inv_chol = self.calc_chol(C_inv)
		C = self.inverse(C_inv_chol)
		gamma = self.calc_gamma(y, m_const)

		mu = np.dot(C, gamma)
		
		mean = m_xstar + np.dot(kx_star.T, np.dot(kx_inv, mu))

		kstar_star = self.kernel_hyper(xprime, xprime, theta_d, 1.0)
		Lambda_chol = self.calc_chol(Lambda)
		Lambda_inv = self.inverse(Lambda_chol)
		kx_lamdainv = kx + Lambda_inv
		kx_lamdainv_chol = self.calc_chol(kx_lamdainv)
		kx_lamdainv_inv = self.inverse(kx_lamdainv_chol)
		cov= kstar_star - np.dot(kx_star.T, np.dot(kx_lamdainv_inv, kx_star))
	
	def predict_new_point1(self, t, tprime, yn, mu_n=None, Cnn=None):
		ktn = self.kernel_curve(t, t, 1.0, 1.0)
		ktn_chol = self.calc_chol(ktn)
		ktn_inv = self.inverse(ktn_chol)
		ktn_star = self.kernel_curve(t, tprime, 1.0, 1.0)
		Omega = np.ones((tprime.shape[0], 1)) - np.dot(ktn_star.T, np.dot(ktn_inv, np.ones((t.shape[0], 1))))
		mean = np.dot(ktn_star.T, np.dot(ktn_inv, yn)) + np.dot(Omega, mu_n)
		ktn_star_star = self.kernel_curve(tprime, tprime, 1.0, 1.0)
		cov = ktn_star_star - np.dot(ktn_star.T, np.dot(ktn_inv, ktn_star)) + np.dot(Omega, np.dot(Cnn, Omega.T))
	
		
	def predict_new_point2(self, one_step_pro_config, x, xprime, y):
		mean_star, Sigma_star_star = self.predict_asy(x, xprime, y)
	
	def getYvector(self, y):
		"""
		Transform the y_train from type ndarray(N, dtype=object) to ndarray(T, 1).
		That's necessary for doing matrices operations
		
		Returns
		-------
		y_vec: ndarray(T,1)
			An array containing all loss measurements of all training curves. They need
			to be stacked in the same order as in the configurations array x_train
		"""
		y_vec = np.array([y[0]])
		for i in xrange(1, y.shape[0]):
			y_vec = np.append(y_vec, y[i])
		return y_vec.reshape(-1, 1)
	
	def nplog(self, val, minval=0.0000000001):
		return np.log(val.clip(min=minval)).reshape(-1, 1)
		
	
	def getOmicron(self, y):
		"""
		Caculates the matrix O = blockdiag(1_1, 1_2,...,1_N), a block-diagonal matrix, where each block is a vector of ones
		corresponding to the number of observations in its corresponding training curve
		
		Parameters
		----------
		y: ndarray(N, dtype=object)
		   All training curves stacked together
		
		Returns
		-------
		O: ndarray(T,N)
			Matrix O is used in several computations in the BO framework, specially in the marginal likelihood
		"""
		O = block_diag(np.ones((y[0].shape[0], 1)))

		for i in xrange(1, y.shape[0]):
			O = block_diag(O, np.ones((y[i].shape[0], 1)))
		return O
	
	def marginal_likelihood(self, theta):
		"""
		Calculates the marginal_likelikood for both the GP over hyperparameters and the GP over the training curves
		
		Parameters
		----------
		theta: all GP hyperparameters
		
		Results
		-------
		marginal likelihood: float
			the resulting marginal likelihood
		"""	
		
		x=self.x_train
		y=self.y_train
		
		flex = self.x_train.shape[-1]

		theta_d = np.zeros(flex)
		theta_d = theta[:flex]
		if not self.samenoise:
			theta0, alpha, beta, noiseHyper, noiseCurve = theta[flex:]
		else:

			theta0, alpha, beta, noise = theta[flex:]
			noiseHyper = noiseCurve = noise

		theta_d = np.exp(theta_d)

		
		self.noiseHyper = exp(noiseHyper)
		
		self.noiseCurve = exp(noiseCurve)
		
		self.theta_d =theta_d

		self.theta0 = np.exp(theta0)
		self.alpha = np.exp(alpha)
		self.beta= np.exp(beta)

		self.m_const = self.get_mconst()
		
		y_vec = self.getYvector(y)
		self.y_vec = y_vec
		#print 'y_vec: ', y_vec.shape
		O = self.getOmicron(y)
		#print 'O: ', O.shape
		kx = self.kernel_hyper(x, x, theta_d, theta0)
		
		if kx == None:
			print 'failed: kx'
			return -np.inf
		#print 'kx: ', kx.shape
		
		#Lambda, gamma = self.lambdaGamma(y, self.m_const)
		Lambda, gamma = self.lambdaGamma2(self.m_const)
		if Lambda == None or gamma == None:
			#print 'failed: lambda or gamma'
			return -np.inf
			

		kx_inv = self.invers(kx)
		if kx_inv==None:
			print 'failed: kx_inv'
			return -np.inf
		#print 'kx_inv: ', kx_inv.shape
		
		kx_inv_plus_L = kx_inv + Lambda
		#print 'kx_inv_plus_L: ', kx_inv_plus_L.shape
		
		kx_inv_plus_L_inv = self.invers(kx_inv_plus_L)
		if kx_inv_plus_L_inv == None:
			print 'failed: kx_inv_plus_L_inv'
			return -np.inf
			
		kt = self.getKt(y)
		
		if kt == None:
			print 'failed: kt'
			return -np.inf
			

		kt_inv = self.invers(kt)
		if kt_inv == None:
			#print 'failed: kt_inv'
			return -np.inf
		
		#print 'y_vec: ', y_vec.shape
		#print 'O: ', O.shape
		#print 'm_const: ', self.m_const.shape
		#print 'kt_inv: ', kt_inv.shape
		#y_minus_Om = y_vec - O*self.m_const
		y_minus_Om = y_vec - np.dot(O, self.m_const)
		
		#print 'np.dot(y_minus_Om.T, np.dot(kt_inv, y_minus_Om)): ', np.dot(y_minus_Om.T, np.dot(kt_inv, y_minus_Om))
		#print 'np.dot(gamma.T, np.dot(kx_inv_plus_L_inv, gamma)): ', np.dot(gamma.T, np.dot(kx_inv_plus_L_inv, gamma))
		#print 'self.nplog(np.linalg.det(kx)): ', self.nplog(np.linalg.det(kx))
		kt = kt/1000.
		#print 'self.nplog(np.linalg.det(kt)): ', self.nplog(np.linalg.det(kt))
		#print 'self.nplog(np.linalg.det(kx_inv_plus_L)): ', self.nplog(np.linalg.det(kx_inv_plus_L))
		logP = -(1/2.)*np.dot(y_minus_Om.T, np.dot(kt_inv, y_minus_Om)) + (1/2.)*np.dot(gamma.T, np.dot(kx_inv_plus_L_inv, gamma))\
		       - (1/2.)*(self.nplog(np.linalg.det(kx_inv_plus_L)) + self.nplog(np.linalg.det(kx)) + self.nplog(np.linalg.det(kt))) # + const #* Where does const come from?
		
		if logP == None or str(logP) == str(np.nan):
			print 'failed: logP' 
			return -np.inf
		#print 'logP: ', logP
		#print 'self.prob_uniform(theta_d): ', self.prob_uniform(theta_d)
		#print 'self.prob_norm(np.array([theta0, alpha, beta])): ', self.prob_norm(np.array([theta0, alpha, beta]))
		#print 'self.prob_horse(np.array([self.noiseHyper, self.noiseCurve])): ', self.prob_horse(np.array([self.noiseHyper, self.noiseCurve]))

		if not self.horse:
			lp = logP + self.prob_uniform(theta_d) + self.prob_norm(np.array([theta0, alpha, beta])) + self.prob_noise(np.array([self.noiseHyper, self.noiseCurve]))# + self.prob_uniform_mconst(m_const)
		else:
			lp = logP + self.prob_uniform(theta_d) + self.prob_norm(np.array([theta0, alpha, beta])) + self.prob_horse(np.array([self.noiseHyper, self.noiseCurve]))

		if lp == None or str(lp) == str(np.nan):
			print 'failed: lp'
			return -np.inf
		
		#print 'lp: ', lp
		return lp
	
	
	def prob_norm(self, theta):
		"""
		Calculates the log probability of samples extracted from the lognormal distribution
		
		Parameters
		----------
		theta: the GP hyperparameters which were drawn from the lognormal distribution
		
		Returns
		-------
		log probability: float
			The sum of the log probabilities of all different samples extracted from the lognorm
		"""
		std = np.zeros_like(theta)
		std[:] = 1.
		probs = sps.lognorm.logpdf(theta, std, loc=np.zeros_like(theta))
		#probs = np.log(sps.lognorm.logpdf(theta, std, loc=np.zeros_like(theta)))
		return np.sum(probs)
	
	def prob_horse(self, theta, scale=0.1):
		if np.any(theta == 0.0):
			#return np.inf
			return -np.inf
		
		#return np.log(np.log(1 + 3.0 * (scale / np.exp(theta)) ** 2))
		return np.sum(np.log(np.log(1 + 3.0 * (scale / np.exp(theta)) ** 2)))

	def prob_noise(self, theta):
		"""
		Calculates the log probability of noise samples extracted from the lognormal distribution
		
		Parameters
		----------
		theta: the GP noise hyperparameters which were drawn from the lognormal distribution
		
		Returns
		-------
		log probability: float
			The sum of the log probabilities of all different noise samples
		"""
		std = np.zeros_like(theta)
		std[:] = 1.
		probs = sps.lognorm.logpdf(theta, std, loc=np.zeros_like(theta))
		return np.sum(probs)

#I'm not sure about this probs[:] = 0.1. I concluded that from some theory, but I should verify it	
	def prob_uniform(self, theta):
		"""
		Calculates the uniform probability of samples extracted from the uniform distribution between 0 and 10
		
		Parameters
		----------
		theta: the GP hyperparameters which were drawn from the uniform distribution between 0 and 10
		
		Returns
		-------
		uniform probability: float
			The sum of the log probabilities of all different samples extracted from the uniform distribution
		"""
		if np.any(theta < 0) or np.any(theta>10):
			return -np.inf
		else:
			probs = np.zeros_like(theta)
			probs[:] = 0.1
			return np.sum(np.log(probs))
	
	def prob_uniform_mconst(self, theta):
		"""
		Calculates the uniform probability of samples extracted from the uniform distribution between y_min and y_max
		
		Parameters
		----------
		theta: the GP hyperparameters which were drawn from the uniform distribution between y_min and y_max
		
		Returns
		-------
		uniform probability: float
			The sum of the log probabilities of all different samples extracted from the uniform distribution
		"""
		mini = np.min(self.y_vec)
		maxi = np.max(self.y_vec)
		if np.any(theta < mini) or np.any(theta>maxi):
			return -np.inf
		else:
			probs = np.zeros_like(theta)
			probs[:] = 1./(maxi-mini)
			return np.sum(np.log(probs))

"""
Based on equation 19 from the freeze-thawn paper
"""

class PredictiveHyper(object):
	"""
	Class for the GP over hyperparameters. The Posterior Predictive Distribution.
	
	Parameters
	----------
	x_train: ndarray(N,D)
		The input training data for all GPs
	y_train: ndarray(T,N)
		The target training data for all GPs
	x_test: ndarray(*,D)
		The current test data for the GPs
	alpha: float
		Hyperparameter of the GP over training curves
	beta: float
		Hyperparameter of the GP over training curves
	theta0: float
		Hyperparameter of the GP over hyperparameter configurations
	thetad = ndarray(D)
		Hyperparameters of the GP over hyperparameter configurations
	"""
	def __init__(self,
				 x_train = None,
				 y_train = None,
				 x_test = None,
				 alpha = 1.0,
				 beta = 1.0,
				 theta0 = 1.0,
				 thetad = None,
				 invChol=True,
				 samenoise = False,
				 kx=None,
				 kx_inv=None):
	 
		 self.x = x_train
		 print 'y_train.shape: ', y_train.shape
		 self.y = y_train
		 self.xprime = x_test
		 self.alpha = alpha
		 self.beta = beta
		 self.theta0 = theta0
		 if thetad == None or self.x.shape[-1] != thetad.shape[0]:
			 self.thetad = np.ones(self.x.shape[-1])
		 else:
			self.thetad = thetad
		 self.C = None
		 self.mu = None
		 self.m_const = None
		 self.invChol = invChol
		 self.samenoise = samenoise
		 self.kx = None
		 self.kx_inv = None
		 self.Kt_chol = None
		 
	def setGpHypers(self, sample):
		"""
		Sets the gp hyperparameters
		
		Parameters
		----------
		sample: ndarray(Number_GP_hyperparameters, 1)
				One sample from the collection of all samples of GP hyperparameters
		"""
		self.m_const = self.get_mconst()
		flex = self.x.shape[-1]
		self.thetad = np.zeros(flex)
		self.thetad = sample[:flex]

		if not self.samenoise:
			self.theta0, self.alpha, self.beta, self.noiseHyper, self.noiseCurve = sample[flex:]
		else:
			self.theta0, self.alpha, self.beta, noise = sample[flex:]
			self.noiseHyper = self.noiseCurve = noise

	def inverse(self, chol):
		''' 
		Once one already has the cholesky of K, one can use this function for calculating the inverse of K
		
		:param chol: the cholesky decomposition of K
		:return: the inverse of K
		'''
		return solve(chol.T, solve(chol, np.eye(chol.shape[0])))
	
	def calc_chol(self, K):
		error_k = 1e-25
		chol = None
		while(True):
			try:
				Ky = K + error_k*np.eye(K.shape[0]) 
				chol = np.linalg.cholesky(Ky)
				break
			except np.linalg.LinAlgError:
				error_k*=10
		return chol

	def invers(self, K):
		if self.invChol:
			invers = self.inverse_chol(K)
		else:
			try:
				invers = np.linalg.inv(K)
			except:
				invers=None
		
		return invers

#change also here with the once and in all 	
	def inverse_chol(self, K):
		''' 
		Once one already has the cholesky of K, one can use this function for calculating the inverse of K
		
		:param chol: the cholesky decomposition of K
		:return: the inverse of K
		'''
		
		chol = self.calc_chol2(K)
		if chol==None:
			return None
		
		#print 'chol: ', chol
		inve = 0
		error_k = 1e-25
		while(True):
			try:
				choly = chol + error_k*np.eye(chol.shape[0])
				#print 'choly.shape: ', choly.shape
				inve = solve(choly.T, solve(choly, np.eye(choly.shape[0])))
				break
			except np.linalg.LinAlgError:
				error_k*=10
		return inve
	
	def inversing(self, chol):
		inve = 0
		error_k = 1e-25
		once = False
		while(True):
			try:
				if once == True:
					choly = chol + error_k*np.eye(chol.shape[0])
				else:
					choly = chol
					once = True
				
				inve = solve(choly.T, solve(choly, np.eye(choly.shape[0])))
				break
			except np.linalg.LinAlgError:
				error_k*=10
		return inve
		

	def calc_chol2(self, K):
		"""
		Calculates the cholesky decomposition of the positive-definite matrix K
		
		Parameters
		----------
		K: ndarray
		   Its dimensions depend on the inputs x for the inputs. len(K.shape)==2
		
		Returns
		-------
		chol: ndarray(K.shape[0], K.shape[1])
			  The cholesky decomposition of K
		"""
		error_k = 1e-25
		chol = None
		index = 0
		found = True
		while(index < 100):
			try:
				#print 'chol index: ', index
				Ky = K + error_k*np.eye(K.shape[0])
				chol = np.linalg.cholesky(Ky)
				found = True
				break
			except np.linalg.LinAlgError:
				error_k*=10
				found = False
			index+=1
		if found:
			return chol
		else:
			return None

	def calc_chol3(self, K):
		"""
		Calculates the cholesky decomposition of the positive-definite matrix K
		
		Parameters
		----------
		K: ndarray
		   Its dimensions depend on the inputs x for the inputs. len(K.shape)==2
		
		Returns
		-------
		chol: ndarray(K.shape[0], K.shape[1])
			  The cholesky decomposition of K
		"""
		error_k = 1e-25
		chol = None
		once = False
		index = 0
		found = True
		while(index < 100):
			try:
				#print 'chol index: ', index
				if once == True:
					Ky = K + error_k*np.eye(K.shape[0])
				else:
					Ky = K
					once = True
				chol = np.linalg.cholesky(Ky)
				found = True
				break
			except np.linalg.LinAlgError:
				error_k*=10
				found = False
			index+=1
		if found:
			return chol
		else:
			return None
	
	def add_xtrain(self, xtrain):
		if self.x == None:
			self.x = xtrain
		else:
			self.x = np.append(self.x, xtrain)
			
	def add_ytrain(self, ytrain):
		if self.y == None:
			self.y = ytrain
		else:
			self.y = np.append(self.y, ytrain)
			
	def add_xtest(self, xtest):
		if self.xprime == None:
			self.xprime = xtest
		else:
			self.xprime = np.append(self.xprime, xtest)
	
	
	def get_mconst(self):
		m_const = np.zeros((len(self.y), 1))
		for i in xrange(self.y.shape[0]):
			mean_i = np.mean(self.y[i], axis=0)
			m_const[i,:] = mean_i
		
		return m_const
	
	def kernel_curve(self, t, tprime):
		"""
		Calculates the kernel for the GP over training curves
		
		Parameters
		----------
		t: ndarray
			learning curve steps
		tprime: ndarray
			learning curve steps. They could be the same or different than t, depending on which covariace is being built
		
		Returns
		-------
		ndarray
			The covariance of t and tprime
		"""

		cov = np.power(self.beta, self.alpha)/np.power(((t[:,np.newaxis] + tprime) + self.beta), self.alpha)
		cov = cov + np.eye(cov.shape[0])*self.noiseCurve
		return cov
		
	def kernel_curve2(self, t, tprime):
		"""
		Calculates the kernel for the GP over training curves
		
		Parameters
		----------
		t: ndarray
			learning curve steps
		tprime: ndarray
			learning curve steps. They could be the same or different than t, depending on which covariace is being built
		
		Returns
		-------
		ndarray
			The covariance of t and tprime
		"""

		try:
			result = np.power(self.beta, self.alpha)/np.power(((t[:,np.newaxis] + tprime) + self.beta), self.alpha)
			result = result + np.eye(result.shape[0])*self.noiseCurve
			return result
		except:
			return None
		
	def kernel_hyper(self, x, xprime):
		"""
		Calculates the kernel for the GP over configuration hyperparameters
		
		Parameters
		----------
		x: ndarray
			Configurations of hyperparameters, each one of shape D
		xprime: ndarray
			Configurations of hyperparameters. They could be the same or different than x, 
			depending on which covariace is being built
		
		Returns
		-------
		ndarray
			The covariance of x and xprime
		"""

		r2 = np.sum(((x[:, np.newaxis] - xprime)**2)/self.thetad, axis=-1)
		fiveR2 = 5*r2
		cov = self.theta0*(1 + np.sqrt(fiveR2) + (5/3.)*fiveR2)*np.exp(-np.sqrt(fiveR2))
		cov = cov + np.eye(cov.shape[0])*self.noiseHyper
		return cov
		
	def kernel_hyper2(self, x, xprime):
		"""
		Calculates the kernel for the GP over configuration hyperparameters
		
		Parameters
		----------
		x: ndarray
			Configurations of hyperparameters, each one of shape D
		xprime: ndarray
			Configurations of hyperparameters. They could be the same or different than x, 
			depending on which covariace is being built
		
		Returns
		-------
		ndarray
			The covariance of x and xprime
		"""

		if len(xprime.shape)==1:
			xprime = xprime.reshape(1,len(xprime))
		if len(x.shape)==1:
			x = x.reshape(1,len(x))
		try:	
			r2 = np.sum(((x[:, np.newaxis] - xprime)**2)/self.thetad**2, axis=-1)
			fiveR2 = 5*r2
			result = self.theta0*(1 + np.sqrt(fiveR2) + (5/3.)*fiveR2)*np.exp(-np.sqrt(fiveR2))
			result = result + np.eye(N=result.shape[0], M=result.shape[1])*self.noiseHyper
			return result
		except:
			return None
	
	def getKt(self, y):
		"""
		Caculates the blockdiagonal covariance matrix Kt. Each element of the diagonal corresponds
		to a covariance matrix Ktn
		
		Parameters
		----------
		y: ndarray(N, dtype=object)
		   All training curves stacked together
		
		Returns
		-------

		"""
		
		ktn = self.getKtn(y[0])
		O = block_diag(ktn)
		
		for i in xrange(1, len(y)):
			ktn = self.getKtn(y[i])
			O = block_diag(O, ktn)
		return O
	
	def getKtn(self, yn):
		t = np.arange(1, yn.shape[0]+1)
		#not yet using the optimized parameters here
		ktn = self.kernel_curve2(t, t)
		#It's already returning None when necessary
		return ktn
		

	def calc_Lambda(self):
		'''
		Calculates Lambda according to the following: Lamda = transpose(O)*inverse(Kt)*O
		= diag(l1, l2,..., ln)=, where ln = transpose(1n)*inverse(Ktn)*1n
		
		Returns
		-------
		Lambda: ndarray(N, N)
				Lamda is used in several calculations in the BO framework
		'''
		dim = self.y.shape[0]
		Lambda = np.zeros((dim, dim))
		index = 0
		for yn in self.y:
			t = np.arange(1, yn.shape[0]+1)
			#not yet using the optimized parameters here
			ktn = self.kernel_curve2(t, t)
			if ktn == None:
				return None
			ktn_inv = self.invers(ktn)
			if ktn_inv == None:
				return None
			one_n = np.ones((ktn.shape[0], 1))
			Lambda[index, index] = np.dot(one_n.T, np.dot(ktn_inv, one_n))
			index+=1
		
		return Lambda
	
	def lambdaGamma(self, m_const):
		dim = self.y.shape[0]
		Lambda = np.zeros((dim, dim))
		gamma = np.zeros((dim, 1))
		for i, yn in enumerate(self.y):
			t = np.arange(1, yn.shape[0]+1)
			ktn = self.kernel_curve2(t, t)
			if ktn == None:
				return None, None
			ktn_inv = self.invers(ktn)
			if ktn_inv == None:
				return None, None
			one_n = np.ones((ktn.shape[0], 1))
			Lambda[i, i] = np.dot(one_n.T, np.dot(ktn_inv, one_n))
			gamma[i, 0] = np.dot(one_n.T, np.dot(ktn_inv, yn - m_const[i]))
		
		return Lambda, gamma

	def lambdaGamma2(self, m_const):
		"""
		Difference here is that the cholesky decomposition is calculated just once for the whole Kt and thereafter
		we solve the linear system for each Ktn.
		"""
		Kt = self.getKt(self.y)
		#print 'Kt.shape: ', Kt.shape
		self.Kt_chol = self.calc_chol3(Kt)
		if self.Kt_chol == None:
			return None, None
		dim = self.y.shape[0]
		Lambda = np.zeros((dim, dim))
		gamma = np.zeros((dim, 1))
		index = 0
		for i, yn in enumerate(self.y):
			lent = yn.shape[0]
			ktn_chol = self.Kt_chol[index:index+lent, index:index+lent]
			#print 'ktn_chol.shape: ', ktn_chol.shape
			index+=lent
			ktn_inv = self.inversing(ktn_chol)
			if ktn_inv == None:
				return None, None
			one_n = np.ones((ktn_inv.shape[0], 1))
			#print 'one_n.shape: ', one_n.shape
			Lambda[i, i] = np.dot(one_n.T, np.dot(ktn_inv, one_n))
			gamma[i, 0] = np.dot(one_n.T, np.dot(ktn_inv, yn - m_const[i]))
		
		return Lambda, gamma
		
		
	def calc_gamma(self, m_const):
		'''
        Calculates gamma according to the following: gamma = transpose(O)*inverse(Kt)*(y - Om),
		where each gamma element gamma_n = transpose(1n)*inverse(Ktn)*(y_n -m_n)
		
		Parameters
		----------
		m_const: float
			the infered mean of f, used in the joint distribution of f and y.
		
		Returns
		-------
		gamma: ndarray(N, 1)
			gamma is used in several calculations in the BO framework
		'''
		dim = self.y.shape[0]
		gamma = np.zeros((dim, 1))
		index = 0
		for i, yn in enumerate(self.y):
			t = np.arange(1, yn.shape[0]+1)
			#not yet using the optimized parameters here
			ktn = self.kernel_curve2(t, t)
			if ktn == None:
				return None
			ktn_inv = self.invers(ktn)
			if ktn_inv == None:
				return None
			one_n = np.ones((ktn.shape[0], 1))
			gamma[index, 0] = np.dot(one_n.T, np.dot(ktn_inv, yn - m_const[i]))
			index+=1
	
		return gamma    	
	

	def predict_asy(self, xprime=None):
		'''
		Given new configuration xprime, it predicts the probability distribution of
		the new asymptotic mean, with mean and covariance of the distribution
		
		Parameters
		----------
		xprime: ndarray(number_configurations, D)
			The new configurations, of which the mean and the std2 are being predicted
		
		Returns
		-------
		mean: ndarray(len(xprime))
			predicted means for each one of the test configurations
		std2: ndarray(len(xprime))
			predicted std2s for each one of the test configurations
		C: ndarray(N,N)
			The covariance of the posterior distribution. It is used several times in the BO framework
		mu: ndarray(N,1)
			The mean of the posterior distribution. It is used several times in the BO framework
		'''
		
		if xprime != None:
			self.xprime = xprime

		theta_d = np.ones(self.x.shape[-1])

		kx_star = self.kernel_hyper2(self.x, self.xprime)
		
		if kx_star == None:
			return None
		
		kx = self.kernel_hyper2(self.x, self.x)
		if kx == None:
			return None
			
		if len(xprime.shape) > 1:
			m_xstar = self.xprime.mean(axis=1).reshape(-1, 1)
		else:
			m_xstar = self.xprime

		m_xstar = np.zeros(m_xstar.shape)

		m_const = self.m_const
		
		
		kx_inv = self.invers(kx)
		if kx_inv == None:
			return None
		m_const = self.m_const
		#print 'm_const.shape: ', m_const.shape
		Lambda, gamma = self.lambdaGamma2(m_const)
		#print 'Lambda.shape: ', Lambda
		if Lambda == None or gamma == None:
			return None
		
		

		C_inv = kx_inv + Lambda
		
		C = self.invers(C_inv)
		if C == None:
			return None
		
		self.C = C
		
		
		mu = self.m_const + np.dot(C, gamma)
		self.mu = mu
		
		mean = m_xstar + np.dot(kx_star.T, np.dot(kx_inv, mu - self.m_const))
		
		#Now calculate the covariance
		kstar_star = self.kernel_hyper2(self.xprime, self.xprime)
		if kstar_star == None:
			return None
		
		Lambda_inv = self.invers(Lambda)
		if Lambda_inv == None:
			return None
		
		kx_lamdainv = kx + Lambda_inv
		

		kx_lamdainv_inv = self.invers(kx_lamdainv)
		if kx_lamdainv_inv == None:
			return None

		cov= kstar_star - np.dot(kx_star.T, np.dot(kx_lamdainv_inv, kx_star))
		std2 = np.diagonal(cov).reshape(-1, 1)

		return mean, std2, C, mu

'''
Based on the equation 20 from freeze-thawn paper
'''

class PredictiveOld(object):
	
	def __init__(self,
				 x_train = None,
				 y_train = None,
				 x_test = None,
				 alpha = None,
				 beta = None,
				 theta0 = None,
				 theta_d = None,
				 invChol = True,
				 samenoise = False):
	 
		 self.x = x_train
		 self.y = y_train
		 self.xprime = x_test
		 self.alpha = alpha
		 self.beta = beta
		 self.theta0 = theta0
		 #if theta_d == None or self.x.shape[-1] != len(theta_d):
			 #self.theta_d = np.ones(self.x.shape[-1])
		 #else:
			#self.theta_d = theta_d
		 self.theta_d = theta_d
		 self.invChol = invChol
		 self.samenoise = samenoise
	
	def setGpHypers(self, sample):
		self.m_const = 0.
		flex = self.x.shape[-1]
		self.thetad = np.zeros(flex)
		self.thetad = sample[:flex]
		if not self.samenoise:
			self.theta0, self.alpha, self.beta, self.noiseHyper, self.noiseCurve = sample[flex:]
		else:
			self.theta0, self.alpha, self.beta, noise = sample[flex:]
			self.noiseHyper = self.noiseCurve = noise	

	def inverse(self, chol):
		''' 
		Once one already has the cholesky of K, one can use this function for calculating the inverse of K
		
		:param chol: the cholesky decomposition of K
		:return: the inverse of K
		'''
		return solve(chol.T, solve(chol, np.eye(chol.shape[0])))
	
	def invers(self, K):
		if self.invChol:
			invers = self.inverse_chol(K)
		else:
			try:
				invers = np.linalg.inv(K)
			except:
				invers=None
		
		return invers
	
	def inverse_chol(self, K):
		''' 
		Once one already has the cholesky of K, one can use this function for calculating the inverse of K
		
		:param chol: the cholesky decomposition of K
		:return: the inverse of K
		'''
		
		chol = self.calc_chol(K)
		if chol==None:
			return None
		
		#print 'chol: ', chol
		inve = 0
		error_k = 1e-25
		while(True):
			try:
				choly = chol + error_k*np.eye(chol.shape[0])
				#print 'choly.shape: ', choly.shape
				inve = solve(choly.T, solve(choly, np.eye(choly.shape[0])))
				break
			except np.linalg.LinAlgError:
				error_k*=10
		return inve	
	
	
	def calc_chol(self, K):
		"""
		Calculates the cholesky decomposition of the positive-definite matrix K
		
		Parameters
		----------
		K: ndarray
		   Its dimensions depend on the inputs x for the inputs. len(K.shape)==2
		
		Returns
		-------
		chol: ndarray(K.shape[0], K.shape[1])
			  The cholesky decomposition of K
		"""

		error_k = 1e-25
		chol = None
		index = 0
		found = True
		while(index < 100):
			try:
				Ky = K + error_k*np.eye(K.shape[0])
				chol = np.linalg.cholesky(Ky)
				found = True
				break
			except np.linalg.LinAlgError:
				error_k*=10
				found = False
			index+=1
			#print index
		if found:
			return chol
		else:
			return None
				
	def calc_chol2(self, K):
		error_k = 1e-25
		chol = None
		while(True):
			try:
				Ky = K + error_k*np.eye(K.shape[0])
				chol = np.linalg.cholesky(Ky)
				break
			except np.linalg.LinAlgError:
				error_k*=10
		return chol
	
	
	def get_mconst(self):
		m_const = np.zeros((len(self.y), 1))
		for i in xrange(self.y.shape[0]):
			mean_i = np.mean(self.y[i], axis=0)
			m_const[i,:] = mean_i
		
		return m_const
	
	def kernel_curve(self, t, tprime):
		try:
			result = np.power(self.beta, self.alpha)/np.power(((t[:,np.newaxis] + tprime) + self.beta), self.alpha)
			result = result + np.eye(N=result.shape[0], M=result.shape[1])*self.noiseCurve
			return result
		except:
			return None
		
	def kernel_hyper(self, x, xprime):
		try:
			r2 = np.sum(((x[:, np.newaxis] - xprime)**2)/self.thetad**2, axis=-1)
			fiveR2 = 5*r2
			result = self.theta0*(1 + np.sqrt(fiveR2) + (5/3.)*fiveR2)*np.exp(-np.sqrt(fiveR2))
			result = result + np.eye(result.shape[0])*self.noiseHyper
			return result
		except:
			return None
	
	def calc_Lambda(self):
		'''
		y is an ndarray object with dtype=object, with all learning curves already running
		'''
		dim = self.y.shape[0]
		Lambda = np.zeros((dim, dim))
		index = 0
		for yn in self.y:
			t = np.arange(1, yn.shape[0]+1)
			#not yet using the optimized parameters here
			ktn = self.kernel_curve(t, t)
			chol_ktn = self.calc_chol(ktn)
			ktn_inv = self.inverse(chol_ktn)
			one_n = np.ones((ktn.shape[0], 1))
			Lambda[index, index] = np.dot(one_n.T, np.dot(ktn_inv, one_n))
			index+=1
		
		return Lambda
		
	def calc_gamma(self, m_const):
		'''
		y is an ndarray object with dtype=object, with all learning curves already running
		'''
		dim = self.y.shape[0]
		gamma = np.zeros((dim, 1))
		index = 0
		for yn in self.y:
			t = np.arange(1, yn.shape[0]+1)
			#not yet using the optimized parameters here
			ktn = self.kernel_curve(t, t)
			chol_ktn = self.calc_chol(ktn)
			ktn_inv = self.inverse(chol_ktn)
			one_n = np.ones((ktn.shape[0], 1))
			gamma[index, 0] = np.dot(one_n.T, np.dot(ktn_inv, yn - m_const[index]))
			index+=1
	
		return gamma    	
	
	def predict_asy(self, xprime=None):
		'''
		Given new configuration xprime, it predicts the probability distribution of
		the new asymptotic mean, with mean and covariance of the distribution
		
		:param x: all configurations without the new ones
		:param xprime: new configurations
		:param y: all training curves
		:type y: ndarray(dtype=object)
		:return: mean of the new configurations
		:return: covariance of the new configurations
		'''
		if xprime!=None:
			self.xprime = xprime
		theta_d = np.ones(self.x.shape[-1])
		kx_star = self.kernel_hyper(self.x, self.xprime)
		kx = self.kernel_hyper(self.x, self.x)
		
		m_xstar = self.xprime.mean(axis=1).reshape(-1, 1)
		m_const = self.get_mconst(self.y)
		cholx = self.calc_chol(kx)
		kx_inv = self.inverse(cholx)
		Lambda = self.calc_Lambda(self.y)
		C_inv = kx_inv + Lambda
		C_inv_chol = self.calc_chol(C_inv)
		C = self.inverse(C_inv_chol)
		gamma = self.calc_gamma(m_const)

		mu = np.dot(C, gamma)
		
		mean = m_xstar + np.dot(kx_star.T, np.dot(kx_inv, mu))
		
		#Now calculate the covariance
		kstar_star = self.kernel_hyper(self.xprime, self.xprime)
		Lambda_chol = self.calc_chol(Lambda)
		Lambda_inv = self.inverse(Lambda_chol)
		kx_lamdainv = kx + Lambda_inv
		kx_lamdainv_chol = self.calc_chol(kx_lamdainv)
		kx_lamdainv_inv = self.inverse(kx_lamdainv_chol)
		cov= kstar_star - np.dot(kx_star.T, np.dot(kx_lamdainv_inv, kx_star))

	def predict_new_point1(self, t, tprime, yn, mu_n=None, Cnn=None):
		yn = yn.reshape(-1,1)

		ktn = self.kernel_curve(t, t)
		if ktn == None:
			return None
		#print 'ktn: ', ktn.shape
		
		ktn_inv = self.invers(ktn)
		if ktn_inv == None:
			return None
		
		ktn_star = self.kernel_curve(t, tprime)
		if ktn_star == None:
			return None
		
		Omega = np.ones((tprime.shape[0], 1)) - np.dot(ktn_star.T, np.dot(ktn_inv, np.ones((t.shape[0], 1))))
		
		#Exactly why:
		if yn.shape[0] > ktn_inv.shape[0]:
			yn = yn[:ktn_inv.shape[0]]
		
		mean = np.dot(ktn_star.T, np.dot(ktn_inv, yn)) + np.dot(Omega, mu_n)

		
		#covariance
		ktn_star_star = self.kernel_curve(tprime, tprime)
		#print 'ktn.shape: ', ktn.shape
		#print 'ktn_star.shape: ', ktn_star.shape
		#print 'ktn_star_star.shape: ', ktn_star_star.shape
		if ktn_star_star == None:
			return None
		
		cov = ktn_star_star - np.dot(ktn_star.T, np.dot(ktn_inv, ktn_star)) + np.dot(Omega, np.dot(Cnn, Omega.T))
		std2 = np.diagonal(cov).reshape(-1,1)
		#print cov.shape
		return mean, std2

class PredictiveNew(object):
	
	def __init__(self,
				 x_train = None,
				 y_train = None,
				 x_test = None,
				 asy_mean = None,
				 alpha = 1.0,
				 beta = 1.0,
				 theta0 = 1.0,
				 theta_d = None,
				 samenoise = False):
		
		self.x = x_train
		self.y = y_train
		self.xprime = x_test
		self.asy_mean = asy_mean
		self.alpha = alpha
		self.beta = beta
		self.theta0 = theta0
		if theta_d == None:
			self.theta_d = np.ones(self.x.shape[-1])
		else:
			self.theta_d = theta_d
		
		self.samenoise = samenoise
	
	
	def setGpHypers(self, sample):
		self.m_const = 0.
		flex = self.x.shape[-1]
		self.thetad = np.zeros(flex)
		self.thetad = sample[:flex]
		if not self.samenoise:
			self.theta0, self.alpha, self.beta, self.noiseHyper, self.noiseCurve = sample[flex:]
		else:
			self.theta0, self.alpha, self.beta, noise = sample[flex:]
			self.noiseHyper = self.noiseCurve = noise
		
		
	def inverse(self, chol):
		''' 
		Once one already has the cholesky of K, one can use this function for calculating the inverse of K
		
		:param chol: the cholesky decomposition of K
		:return: the inverse of K
		'''
		return solve(chol.T, solve(chol, np.eye(chol.shape[0])))
	
	def calc_chol(self, K):
		error_k = 1e-25
		chol = None
		while(True):
			try:
				Ky = K + error_k*np.eye(K.shape[0])
				chol = np.linalg.cholesky(Ky)
				break
			except np.linalg.LinAlgError:
				error_k*=10
		return chol

	def calc_chol2(self, K):
		"""
		Calculates the cholesky decomposition of the positive-definite matrix K
		
		Parameters
		----------
		K: ndarray
		   Its dimensions depend on the inputs x for the inputs. len(K.shape)==2
		
		Returns
		-------
		chol: ndarray(K.shape[0], K.shape[1])
			  The cholesky decomposition of K
		"""
		error_k = 1e-25
		chol = None
		index = 0
		found = True
		while(index < 100):
			try:
				Ky = K + error_k*np.eye(K.shape[0])
				chol = np.linalg.cholesky(Ky)
				found = True
				break
			except np.linalg.LinAlgError:
				error_k*=10
				found = False
			index+=1
		if found:
			return chol
		else:
			return None
	
	
	def get_mconst(self):
		m_const = np.zeros((len(self.y), 1))
		for i in xrange(self.y.shape[0]):
			mean_i = np.mean(self.y[i], axis=0)
			m_const[i,:] = mean_i
		
		return m_const
	
	def kernel_curve(self, t, tprime):
		return np.power(self.beta, self.alpha)/np.power(((t[:,np.newaxis] + tprime) + self.beta), self.alpha)
		
	def kernel_hyper(self, x, xprime):
		r2 = np.sum(((x[:, np.newaxis] - xprime)**2)/self.thetad, axis=-1)
		fiveR2 = 5*r2
		return  self.theta0*(1 + np.sqrt(fiveR2) + (5/3.)*fiveR2)*np.exp(-np.sqrt(fiveR2))
	
	def calc_Lambda(self):
		'''
		y is an ndarray object with dtype=object, with all learning curves already running
		'''
		dim = self.y.shape[0]
		Lambda = np.zeros((dim, dim))
		index = 0
		for yn in self.y:
			t = np.arange(1, yn.shape[0]+1)
			#not yet using the optimized parameters here
			ktn = self.kernel_curve(t, t)
			chol_ktn = self.calc_chol(ktn)
			ktn_inv = self.inverse(chol_ktn)
			one_n = np.ones((ktn.shape[0], 1))
			Lambda[index, index] = np.dot(one_n.T, np.dot(ktn_inv, one_n))
			index+=1
		
		return Lambda
		
	def calc_gamma(self, m_const):
		'''
		y is an ndarray object with dtype=object, with all learning curves already running
		'''
		dim = self.y.shape[0]
		gamma = np.zeros((dim, 1))
		index = 0
		for yn in self.y:
			t = np.arange(1, yn.shape[0]+1)
			#not yet using the optimized parameters here
			ktn = self.kernel_curve(t, t)
			chol_ktn = self.calc_chol(ktn)
			ktn_inv = self.inverse(chol_ktn)
			one_n = np.ones((ktn.shape[0], 1))
			gamma[index, 0] = np.dot(one_n.T, np.dot(ktn_inv, yn - m_const[index]))
			index+=1
	
		
		return gamma    	
	
	def predict_asy(self, xprime=None):
		'''
		Given new configuration xprime, it predicts the probability distribution of
		the new asymptotic mean, with mean and covariance of the distribution
		
		:param x: all configurations without the new ones
		:param xprime: new configurations
		:param y: all training curves
		:type y: ndarray(dtype=object)
		:return: mean of the new configurations
		:return: covariance of the new configurations
		'''
		if xprime != None:
			self.xprime = xprime
		theta_d = np.ones(self.x.shape[-1])
		kx_star = self.kernel_hyper(self.x, self.xprime)
		kx = self.kernel_hyper(self.x, self.x)

		
		if len(self.xprime.shape)==2:
			m_xstar = np.zeros(self.xprime.shape[0])
		else:
			m_xstar = np.zeros(1) 
		
		m_const = self.get_mconst()
		cholx = self.calc_chol(kx)
		kx_inv = self.inverse(cholx)
		Lambda = self.calc_Lambda()
		C_inv = kx_inv + Lambda
		C_inv_chol = self.calc_chol(C_inv)
		C = self.inverse(C_inv_chol)
		gamma = self.calc_gamma(m_const)

		mu = m_const + np.dot(C, gamma)	
		
		mean = m_xstar + np.dot(kx_star.T, np.dot(kx_inv, (mu - m_const)))
		
		#Now calculate the covariance
		kstar_star = self.kernel_hyper(self.xprime, self.xprime)
		Lambda_chol = self.calc_chol(Lambda)
		Lambda_inv = self.inverse(Lambda_chol)
		kx_lamdainv = kx + Lambda_inv
		kx_lamdainv_chol = self.calc_chol(kx_lamdainv)
		kx_lamdainv_inv = self.inverse(kx_lamdainv_chol)
		cov= kstar_star - np.dot(kx_star.T, np.dot(kx_lamdainv_inv, kx_star))
		return mean, cov
	
	def predict_new_point1(self, t, tprime, yn, mu_n=None, Cnn=None):
		ktn = self.kernel_curve(t, t)
		ktn_chol = self.calc_chol(ktn)
		ktn_inv = self.inverse(ktn_chol)
		ktn_star = self.kernel_curve(t, tprime)
		Omega = np.ones((tprime.shape[0], 1)) - np.dot(ktn_star.T, np.dot(ktn_inv, np.ones((t.shape[0], 1))))
		mean = np.dot(ktn_star.T, np.dot(ktn_inv, yn)) + np.dot(Omega, mu_n)
		
		#covariance
		ktn_star_star = self.kernel_curve(tprime, tprime)
		cov = ktn_star_star - np.dot(ktn_star.T, np.dot(ktn_inv, ktn_star)) + np.dot(Omega, np.dot(Cnn, Omega.T))
	
		
	
	def predict_new_point2(self, step, asy_mean, y=None):
		if y!=None:
			self.y = y

		if asy_mean != None:
			self.asy_mean = asy_mean
		fro = 1
		t = np.arange(fro,(fro+1))
		tprime = np.arange((fro+1),(fro+1)+step)
		k_xstar_x = self.kernel_curve(tprime, t)
		k_x_x = self.kernel_curve(t, t)

		chol = self.calc_chol2(k_x_x + self.noiseCurve*np.eye(k_x_x.shape[0]))
		
		#Exactly why:
		self.y=np.array([1.])
		sol = np.linalg.solve(chol, self.y)
		sol = np.linalg.solve(chol.T, sol)
		
		k_xstar_xstar = self.kernel_curve(tprime, tprime)
		k_x_xstar = k_xstar_x.T
		mean = self.asy_mean + np.dot(k_xstar_x, sol)
		solv = np.linalg.solve(chol, k_x_xstar)
		solv = np.dot(solv.T, solv)
		cov = k_xstar_xstar - solv
		std2 = np.diagonal(cov).reshape(-1, 1)
		return mean, std2
