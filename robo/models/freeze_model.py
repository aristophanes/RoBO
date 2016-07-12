# encoding=utf8
__author__ = "Tulio Paiva"
__email__ = "paivat@cs.uni-freiburg.de"

#from GPy import Model, Param
#from GPy.kern import Kern
#import GPy
#import cython
import scipy
import numpy as np
from robo.util.freezeProcess.predictiveLikelihood import PredLik
from robo.util.freezeProcess.LikIntegrate import LikIntegrate
import logging
from scipy import optimize
from robo.models.base_model import BaseModel
import robo.util.freezeProcess.PredManySamples as PredManySamples
import robo.util.freezeProcess.LikIntegrate as LikIntegrate
from robo.util.freezeProcess.PredictiveHyper import PredictiveHyper
from sklearn.metrics import mean_squared_error as mse
import time
import matplotlib.pyplot as pl

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
			self.ps = PredManySamples.PredSamples(samples=sampleSet, x_train = self.X, y_train = self.y, x_test = self.X, predHyper=True, predOld=True, predNew=True, invChol=True, samenoise=True)
	
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
			lik = LikIntegrate.LikIntegrate(y_train=self.y, invChol = True, horse=True, samenoise=True)
			sampleSet = lik.create_configs(x_train=self.X, y_train=self.y, hyper_configs=12, chain_length=100, burnin_steps=100)
			
			self.ps = PredManySamples.PredSamples(samples=sampleSet, x_train = self.X, y_train = self.y, x_test = self.X, predHyper=True, predOld=True, predNew=True, invChol=True, samenoise=True)
		
	
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
