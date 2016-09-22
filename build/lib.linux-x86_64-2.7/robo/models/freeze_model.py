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
from robo.priors.base_prior import BasePrior, TophatPrior, \
LognormalPrior, HorseshoePrior, UniformPrior

logger = logging.getLogger(__name__)

class FreezeThawGP(BaseModel):

    def __init__(self,
                 x_train=None,
                 y_train=None,
                 x_test=None,
                 y_test=None,
                 sampleSet=None, 
                 hyper_configs=12, 
                 chain_length=100, 
                 burnin_steps=100,
                 invChol=True,
                 horse=True, 
                 samenoise=True):
        """
        Interface to the freeze-thawn GP library. The GP hyperparameter are obtained
        by integrating out the marginal loglikelihood over the GP hyperparameters.
        
        Parameters
        ----------
        x_train: ndarray(N,D)
            The input training data for all GPs
        y_train: ndarray(N,T)
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
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        
        self.hyper_configs = hyper_configs
        self.chain_length = chain_length
        self.burnin_steps = burnin_steps
        
        self.invChol=invChol
        self.horse = horse
        self.samenoise = samenoise
        
        self.uPrior = UniformPrior(minv=0, maxv=10)
        self.lnPrior = LognormalPrior(sigma=0.1, mean=0.0)
        self.hPrior = HorseshoePrior()

        self.C_samples = np.zeros(
            (self.hyper_configs, self.x_train.shape[0], self.x_train.shape[0]))
        self.mu_samples = np.zeros(
            (self.hyper_configs, self.x_train.shape[0], 1))
        self.activated = False

    
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

        if X is not None:
            self.X = X
        if Y is not None:
            self.Y = Y

        if do_optimize:
            sampleSet = self.create_configs(x_train=self.X, y_train=self.y, hyper_configs=self.hyper_configs, chain_length=self.chain_length, burnin_steps=self.burnin_steps)
        self.samples = sampleSet

    
    def predict(self, xprime=None, option='asympt', conf_nr=0, from_step=None, further_steps=1):
        """
        Predict using one of thre options: (1) predicion of the asymtote given a new configuration,
        (2) prediction of a new step of an old configuration, (3) prediction of steps of a curve of 
        a completely new configuration

        Parameters
        ----------
        xprime: ndarray(N,D)
            The new configuration(s)
        option: string
            The prediction type: 'asympt', 'old', 'new'
        conf_nr: integer
            The index of an old configuration of which a new step is predicted
        from_step: integer
            The step from which the prediction begins for an old configuration.
            If none is give, it is assumend one is predicting from the last step
        further_steps: integer
            How many steps must be predicted from 'from_step'/last step onwards

        Results
        -------
        return: ndarray(N, steps), ndarray(N, steps)
            Mean and variance of the predictions
        """
        if option == 'asympt':
            mu, std2, _ = self.pred_asympt_all(xprime)
        elif option == 'old': 
            if from_step is None:
                mu, std2, _ = self.pred_old_all(
                    conf_nr=conf_nr + 1, steps=further_steps)
            else:
                mu, std2, _ = self.pred_old_all(
                    conf_nr=conf_nr + 1, steps=further_steps, fro=from_step)
        elif option == 'new':
            mu, std2 = self.pred_new_all(
                steps=further_steps, xprime=xprime, asy=True)

        return mu, std2



    def setGpHypers(self, sample):
        """
        Sets the gp hyperparameters

        Parameters
        ----------
        sample: ndarray(Number_GP_hyperparameters, 1)
            One sample from the collection of all samples of GP hyperparameters
        """
        self.m_const = self.get_mconst()
        flex = self.X.shape[-1]
        self.thetad = np.zeros(flex)
        self.thetad = sample[:flex]

        if not self.samenoise:
            self.theta0, self.alpha, self.beta, self.noiseHyper, self.noiseCurve = sample[flex:]
        else:
            self.theta0, self.alpha, self.beta, noise = sample[flex:]
            self.noiseHyper = self.noiseCurve = noise

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

        # number of length scales
        flex = x_train.shape[-1]

        if not self.samenoise:
            #theta0, noiseHyper, noiseCurve, alpha, beta, m_const
            fix = 5
        else:
            #theta0, noise, alpha, beta, m_const
            fix = 4

        #pdl = PredLik(x_train, y_train, invChol=self.invChol,
        #              horse=self.horse, samenoise=self.samenoise)

        samples = np.zeros((hyper_configs, fix + flex))

        sampler = emcee.EnsembleSampler(
            hyper_configs, fix + flex, self.marginal_likelihood)

        # sample length scales for GP over configs
        #uPrior = UniformPrior(minv=0, maxv=10)
        p0a = self.uPrior.sample_from_prior(n_samples=(hyper_configs, flex))

        # sample amplitude for GP over configs and alpha e beta for GP over
        # curve
        #lnPrior = LognormalPrior(sigma=0.1, mean=0.0)
        p0b = self.lnPrior.sample_from_prior(n_samples=(hyper_configs, 3))

        p0 = np.append(p0a, p0b, axis=1)

        #hPrior = HorseshoePrior()

        if not self.samenoise:
            if not self.horse:
                p0d = self.lnPrior.sample_from_prior(n_samples=(hyper_configs, 2))
            else:
                p0d = np.abs(self.hPrior.sample_from_prior(
                    n_samples=(hyper_configs, 2)))
        else:
            if not self.horse:
                p0d = self.lnPrior.sample_from_prior(n_samples=(hyper_configs, 1))
            else:
                p0d = np.abs(self.hPrior.sample_from_prior(
                    n_samples=(hyper_configs, 1)))

        p0 = np.append(p0, p0d, axis=1)

        p0, _, _ = sampler.run_mcmc(p0, burnin_steps)

        pos, prob, state = sampler.run_mcmc(p0, chain_length)

        p0 = pos

        samples = sampler.chain[:, -1]

        return np.exp(samples)

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

        x = self.x_train
        y = self.y_train

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

        self.theta_d = theta_d

        self.theta0 = np.exp(theta0)
        self.alpha = np.exp(alpha)
        self.beta = np.exp(beta)

        self.m_const = self.get_mconst()

        y_vec = self.getYvector(y)
        self.y_vec = y_vec
        # print 'y_vec: ', y_vec.shape
        O = self.getOmicron(y)
        # print 'O: ', O.shape
        kx = self.kernel_hyper(x, x)

        if kx is None:
            #print 'failed: kx'
            return -np.inf
        # print 'kx: ', kx.shape

        Lambda, gamma = self.lambdaGamma(self.m_const)
        if Lambda is None or gamma is None:
            # print 'failed: lambda or gamma'
            return -np.inf

        kx_inv = self.invers(kx)
        if kx_inv is None:
            #print 'failed: kx_inv'
            return -np.inf
        # print 'kx_inv: ', kx_inv.shape

        kx_inv_plus_L = kx_inv + Lambda
        # print 'kx_inv_plus_L: ', kx_inv_plus_L.shape

        kx_inv_plus_L_inv = self.invers(kx_inv_plus_L)
        if kx_inv_plus_L_inv is None:
            #print 'failed: kx_inv_plus_L_inv'
            return -np.inf

        kt = self.getKt(y)

        if kt is None:
            #print 'failed: kt'
            return -np.inf

        kt_inv = self.invers(kt)
        if kt_inv is None:
            # print 'failed: kt_inv'
            return -np.inf

        # print 'y_vec: ', y_vec.shape
        # print 'O: ', O.shape
        # print 'm_const: ', self.m_const.shape
        # print 'kt_inv: ', kt_inv.shape
        #y_minus_Om = y_vec - O*self.m_const
        y_minus_Om = y_vec - np.dot(O, self.m_const)

        # print 'np.dot(y_minus_Om.T, np.dot(kt_inv, y_minus_Om)): ', np.dot(y_minus_Om.T, np.dot(kt_inv, y_minus_Om))
        # print 'np.dot(gamma.T, np.dot(kx_inv_plus_L_inv, gamma)): ', np.dot(gamma.T, np.dot(kx_inv_plus_L_inv, gamma))
        # print 'self.nplog(np.linalg.det(kx)): ',
        # self.nplog(np.linalg.det(kx))
        kt = kt / 1000.
        # print 'self.nplog(np.linalg.det(kt)): ', self.nplog(np.linalg.det(kt))
        # print 'self.nplog(np.linalg.det(kx_inv_plus_L)): ',
        # self.nplog(np.linalg.det(kx_inv_plus_L))
        logP = -(1 / 2.) * np.dot(y_minus_Om.T, np.dot(kt_inv, y_minus_Om)) + (1 / 2.) * np.dot(gamma.T, np.dot(kx_inv_plus_L_inv, gamma))\
               - (1 / 2.) * (self.nplog(np.linalg.det(kx_inv_plus_L)) + self.nplog(np.linalg.det(kx)
                                                                                   ) + self.nplog(np.linalg.det(kt)))  # + const #* Where does const come from?

        if logP is None or str(logP) == str(np.nan):
            #print 'failed: logP'
            return -np.inf
        # print 'logP: ', logP
        # print 'self.prob_uniform(theta_d): ', self.prob_uniform(theta_d)
        # print 'self.prob_norm(np.array([theta0, alpha, beta])): ', self.prob_norm(np.array([theta0, alpha, beta]))
        # print 'self.prob_horse(np.array([self.noiseHyper, self.noiseCurve])):
        # ', self.prob_horse(np.array([self.noiseHyper, self.noiseCurve]))

        # if not self.horse:
            # lp = logP + self.prob_uniform(theta_d) + self.prob_norm(np.array([theta0, alpha, beta])) + self.prob_noise(np.array([self.noiseHyper, self.noiseCurve]))# + self.prob_uniform_mconst(m_const)
        # else:
            #lp = logP + self.prob_uniform(theta_d) + self.prob_norm(np.array([theta0, alpha, beta])) + self.prob_horse(np.array([self.noiseHyper, self.noiseCurve]))

        #if not self.horse:
            #lp = logP + np.sum(self.uPrior.lnprob(theta_d)) + np.sum(self.lnPrior.lnprob(np.array(
                #[theta0, alpha, beta]))) + np.sum(self.lnPrior.lnprob(np.array([self.noiseHyper, self.noiseCurve])))
        #else:
            #lp = logP + np.sum(self.uPrior.lnprob(theta_d)) + np.sum(self.lnPrior.lnprob(np.array(
                #[theta0, alpha, beta]))) + np.sum(self.hPrior.lnprob(np.array([self.noiseHyper, self.noiseCurve])))
        
        lp = logP + np.sum(self.uPrior.lnprob(theta_d)) + np.sum(self.lnPrior.lnprob(np.array([theta0, alpha, beta]))) + np.sum(self.hPrior.lnprob(np.array([self.noiseHyper, self.noiseCurve])))

        if lp is None or str(lp) == str(np.nan):
            #print 'failed: lp'
            return -np.inf

        # print 'lp: ', lp
        return lp

    def get_mconst(self):
        m_const = np.zeros((len(self.y_train), 1))
        
        for i in xrange(self.y_train.shape[0]):
            mean_i = np.mean(self.y_train[i], axis=0)
            m_const[i, :] = mean_i

        return m_const

    def pred_asympt(self, xprime=None):
        """
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
        """

        if xprime is not None:
            self.xprime = xprime

        theta_d = np.ones(self.X.shape[-1])

        kx_star = self.kernel_hyper(self.X, self.xprime)

        if kx_star is None:
            return None

        kx = self.kernel_hyper(self.X, self.X)
        if kx is None:
            return None

        if len(xprime.shape) > 1:
            m_xstar = self.xprime.mean(axis=1).reshape(-1, 1)
        else:
            m_xstar = self.xprime

        m_xstar = np.zeros(m_xstar.shape)

        m_const = self.m_const

        kx_inv = self.invers(kx)
        if kx_inv is None:
            return None

        m_const = self.m_const
        #print 'm_const.shape: ', m_const.shape
        Lambda, gamma = self.lambdaGamma(m_const)
        #print 'Lambda.shape: ', Lambda
        if Lambda is None or gamma is None:
            return None


        C_inv = kx_inv + Lambda

        C = self.invers(C_inv)
        if C is None:
            return None

        self.C = C

        mu = self.m_const + np.dot(C, gamma)

        self.mu = mu

        mean = m_xstar + np.dot(kx_star.T, np.dot(kx_inv, mu - self.m_const))
        
        #Now calculate the covariance
        kstar_star = self.kernel_hyper(self.xprime, self.xprime)
        if kstar_star is None:
            return None

        Lambda_inv = self.invers(Lambda)
        if Lambda_inv is None:
            return None

        kx_lamdainv = kx + Lambda_inv

        kx_lamdainv_inv = self.invers(kx_lamdainv)

        if kx_lamdainv_inv is None:
            return None

        cov= kstar_star - np.dot(kx_star.T, np.dot(kx_lamdainv_inv, kx_star))
        std2 = np.diagonal(cov).reshape(-1, 1)

        return mean, std2, C, mu

    def pred_asympt_all(self, xprime=None):
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
        if xprime is not None:
            self.x_test = xprime

        samples_val = []
        C_valid = []
        mu_val = []
        means_val = []
        std2s_val = []

        divby = self.samples.shape[0]

        for i in xrange(self.samples.shape[0]):
            self.setGpHypers(self.samples[i])
            pre = self.pred_asympt(xprime)
            # print 'pre: ', pre
            if pre is not None:
                mean_one, std2_one, C, mu = pre
                # print 'mean_one: ', mean_one
                means_val.append(mean_one.flatten())
                # print 'means_val: ', means_val
                std2s_val.append(std2_one.flatten())
                C_valid.append(C)
                mu_val.append(mu)
                samples_val.append(self.samples[i])
            else:
                divby -= 1
                # print 'bad: ', divby

        mean_temp = np.zeros((divby, xprime.shape[0]))
        std2_temp = np.zeros((divby, xprime.shape[0]))

        if(divby < self.samples.shape[0]):
            self.C_samples = np.zeros(
                (divby, self.C_samples.shape[1], self.C_samples.shape[2]))
            self.mu_samples = np.zeros(
                (divby, self.mu_samples.shape[1], self.mu_samples.shape[2]))
            self.samples = np.zeros((divby, self.samples.shape[1]))

        for j in xrange(divby):
            mean_temp[j, :] = means_val[j]
            std2_temp[j, :] = std2s_val[j]
            self.C_samples[j, ::] = C_valid[j]
            self.mu_samples[j, ::] = mu_val[j]
            self.samples[j, ::] = samples_val[j]

        mean = np.mean(mean_temp, axis=0)
        std2 = np.mean(std2_temp, axis=0) + np.mean(mean_temp**2, axis=0)
        std2 -= mean**2

        self.activated = True
        self.asy_mean = mean

        return mean, std2, divby

    def pred_old(self, t, tprime, yn, mu_n=None, Cnn=None):
        yn = yn.reshape(-1, 1)

        ktn = self.kernel_curve(t, t)
        if ktn is None:
            return None
        # print 'ktn: ', ktn.shape

        ktn_inv = self.invers(ktn)
        if ktn_inv is None:
            return None

        ktn_star = self.kernel_curve(t, tprime)
        if ktn_star is None:
            return None

        Omega = np.ones((tprime.shape[0], 1)) - np.dot(ktn_star.T,
                                                       np.dot(ktn_inv, np.ones((t.shape[0], 1))))

        # Exactly why:
        if yn.shape[0] > ktn_inv.shape[0]:
            yn = yn[:ktn_inv.shape[0]]

        mean = np.dot(ktn_star.T, np.dot(ktn_inv, yn)) + np.dot(Omega, mu_n)

        # covariance
        ktn_star_star = self.kernel_curve(tprime, tprime)
        # print 'ktn.shape: ', ktn.shape
        # print 'ktn_star.shape: ', ktn_star.shape
        # print 'ktn_star_star.shape: ', ktn_star_star.shape
        if ktn_star_star is None:
            return None

        cov = ktn_star_star - \
            np.dot(ktn_star.T, np.dot(ktn_inv, ktn_star)) + \
            np.dot(Omega, np.dot(Cnn, Omega.T))
        std2 = np.diagonal(cov).reshape(-1, 1)
        # print cov.shape
        return mean, std2

    def pred_old_all(self, conf_nr, steps, fro=None):
        """
        Here conf_nr is from 1 onwards. That's why we are using mu_n = mu[conf_nr - 1, 0] in the for-loop
        """

        if self.activated:

            means_val = []
            std2s_val = []
            divby = self.samples.shape[0]

            yn = self.y_train[conf_nr - 1]
            if fro is None:
                t = np.arange(1, yn.shape[0] + 1)
                tprime = np.arange(yn.shape[0] + 1, yn.shape[0] + 1 + steps)
            else:
                t = np.arange(1, fro)
                tprime = np.arange(fro, fro + steps)

            for i in xrange(self.samples.shape[0]):
                # print 'samples: ', self.samples
                self.setGpHypers(self.samples[i])

                mu = self.mu_samples[i, ::]
                mu_n = mu[conf_nr - 1, 0]

                C = self.C_samples[i, ::]
                Cnn = C[conf_nr - 1, conf_nr - 1]

                pre = self.pred_old(t, tprime, yn, mu_n, Cnn)

                if pre is not None:
                    mean_one, std2_one = pre
                    means_val.append(mean_one.flatten())
                    std2s_val.append(std2_one.flatten())
                else:
                    divby -= 1

            mean_temp = np.zeros((divby, steps))
            std2_temp = np.zeros((divby, steps))

            for j in xrange(divby):
                mean_temp[j, :] = means_val[j]
                std2_temp[j, :] = std2s_val[j]

            mean = np.mean(mean_temp, axis=0)
            std2 = np.mean(std2_temp, axis=0) + np.mean(mean_temp**2, axis=0)
            std2 -= mean**2

            return mean, std2, divby

        else:
            raise Exception


    def pred_new(self, step, asy_mean, y=None):
        if y is not None:
            self.y = y

        if asy_mean is not None:
            self.asy_mean = asy_mean
        fro = 1
        t = np.arange(fro, (fro + 1))
        tprime = np.arange((fro + 1), (fro + 1) + step)
        k_xstar_x = self.kernel_curve(tprime, t)
        k_x_x = self.kernel_curve(t, t)

        chol = self.calc_chol(
            k_x_x + self.noiseCurve * np.eye(k_x_x.shape[0]))

        # Exactly why:
        self.y = np.array([1.])
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

    def pred_new_all(self, steps=13, xprime=None, y=None, asy=False):
        """
        Params
        ------
        asy: Whether the asymptotic has already been calculated or not.
        """
        if xprime is not None:
            self.x_test = xprime

        # Not redundant here. The  PredictiveHyper object is already created. In case kx has already been calculate
        # it's not going to be calculated a second time.
        if asy is False:
            asy_mean, std2star, _ = self.pred_hyper2(xprime)
        else:
            asy_mean = self.asy_mean

        mean_temp = np.zeros((self.samples.shape[0], steps))
        std2_temp = np.zeros((self.samples.shape[0], steps))

        for i in xrange(self.samples.shape[0]):
            self.setGpHypers(self.samples[i])

            mean_one, std2_one = self.pred_new(
                steps, asy_mean[1], y)
            mean_temp[i, :] = mean_one.flatten()
            std2_temp[i, :] = std2_one.flatten()

        mean = np.mean(mean_temp, axis=0)
        std2 = np.mean(std2_temp, axis=0) + np.mean(mean_temp**2, axis=0)
        std2 -= mean**2
        return mean, std2


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
        # print 'x.shape: ', x.shape
        # print 'xprime.shape: ', xprime.shape
        # print 'theta_d.shape: ', theta_d.shape
        # print 'theta0: ', theta0

        if len(xprime.shape)==1:
            xprime = xprime.reshape(1,len(xprime))

        if len(x.shape)==1:
            x = x.reshape(1,len(x))
        try:
            r2 = np.sum(((x[:, np.newaxis] - xprime)**2) /
                self.theta_d**2, axis=-1)
            # print 'r2: ', r2
            fiveR2 = 5 * r2
            result = self.theta0 *(1 + np.sqrt(fiveR2) + (5 / 3.) * fiveR2)* np.exp(-np.sqrt(fiveR2))
            # print 'result1: ', result
            result = result + \
                np.eye(M=result.shape[0], N=result.shape[1]) * self.noiseHyper
            # print 'result2: ', result
            return result
        except:
            return None

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
            result = np.power(self.beta, self.alpha) / \
                np.power(((t[:, np.newaxis] + tprime) + self.beta), self.alpha)
            # print 'result1 in kernel_curve: ', result
            #result = result + np.eye(result.shape)*self.noiseCurve
            result = result + \
                np.eye(M=result.shape[0], N=result.shape[1]) * self.noiseCurve
            # print 'result2 in kernel_curve: ', result
            return result
        except:
            return None

    def lambdaGamma(self, m_const):
        """
        Difference here is that the cholesky decomposition is calculated just once for the whole Kt and thereafter
        we solve the linear system for each Ktn.
        """
        Kt = self.getKt(self.y)
        # print 'Kt.shape: ', Kt.shape
        self.Kt_chol = self.calc_chol(Kt)
        if self.Kt_chol is None:
            return None, None
        dim = self.y.shape[0]
        Lambda = np.zeros((dim, dim))
        gamma = np.zeros((dim, 1))
        index = 0
        for i, yn in enumerate(self.y):
            lent = yn.shape[0]
            ktn_chol = self.Kt_chol[index:index + lent, index:index + lent]
            # print 'ktn_chol.shape: ', ktn_chol.shape
            index += lent
            ktn_inv = self.inverse_chol(K=None, Chl=ktn_chol)
            if ktn_inv is None:
                return None, None
            one_n = np.ones((ktn_inv.shape[0], 1))
            # print 'one_n.shape: ', one_n.shape
            Lambda[i, i] = np.dot(one_n.T, np.dot(ktn_inv, one_n))
            gamma[i, 0] = np.dot(one_n.T, np.dot(ktn_inv, yn - m_const[i]))

        return Lambda, gamma

    def getKtn(self, yn):
        t = np.arange(1, yn.shape[0] + 1)
        # not yet using the optimized parameters here
        # print 't range: ', t
        ktn = self.kernel_curve(t, t, 1., 1.)
        # It's already returning None when necessary
        return ktn

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
            # print 'in getKt() y[i]: ', y[i]
            ktn = self.getKtn(y[i])
            # print 'in getKt() ktn: ', ktn
            # print
            O = block_diag(O, ktn)
        return O

    def invers(self, K):
        if self.invChol:
            invers = self.inverse_chol(K)
        else:
            try:
                invers = np.linalg.inv(K)
            except:
                invers = None

        return invers

    # def inversing(self, chol):
    #     """
    #     One can use this function for calculating the inverse of K once one has already the
    #     cholesky decompostion

    #     :param chol: the cholesky decomposition of K
    #     :return: the inverse of K
    #     """
    #     inve = 0
    #     error_k = 1e-25
    #     once = False
    #     while(True):
    #         try:
    #             if once is True:
    #                 choly = chol + error_k * np.eye(chol.shape[0])
    #             else:
    #                 choly = chol
    #                 once = True

    #             inve = solve(choly.T, solve(choly, np.eye(choly.shape[0])))
    #             break
    #         except np.linalg.LinAlgError:
    #             error_k *= 10
    #     return inve

    def inverse_chol(self, K=None, Chl=None):
        """ 
        One can use this function for calculating the inverse of K through cholesky decomposition

        Parameters
        ----------
        K: ndarray
            covariance K
        Chl: ndarray
            cholesky decomposition of K

        Returns
        -------
        :return: the inverse of K
        """
        if Chl is not None:
            chol = Chl
        else:
            chol = self.calc_chol(K)

        if chol is None:
            return None

        inve = 0
        error_k = 1e-25
        while(True):
            try:
                choly = chol + error_k * np.eye(chol.shape[0])
                inve = solve(choly.T, solve(choly, np.eye(choly.shape[0])))
                break
            except np.linalg.LinAlgError:
                error_k *= 10
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
        
        # print 'K: ', K
        error_k = 1e-25
        chol = None
        once = False
        index = 0
        found = True
        while(index < 100):
            try:
                # print 'chol index: ', index
                if once is True:
                    # print 'once is True'
                    Ky = K + error_k * np.eye(K.shape[0])
                else:
                    # print 'once is False'
                    #Ky = K
                    Ky = K + error_k * np.eye(K.shape[0])
                    once = True
                chol = np.linalg.cholesky(Ky)
                # print 'chol: ', chol
                found = True
                break
            except np.linalg.LinAlgError:
                # print 'except'
                error_k *= 10
                found = False
            # print 'index: ', index
            index += 1
        if found:
            # print 'it is found'
            return chol
        else:
            # print 'not found'
            return None

    def nplog(self, val, minval=0.0000000001):
        return np.log(val.clip(min=minval)).reshape(-1, 1)
    
