# encoding=utf8
__author__ = "Tulio Paiva"
__email__ = "paivat@cs.uni-freiburg.de"

import numpy as np
from robo.models.freeze_thaw_model3 import FreezeThawGP
from robo.task.synthetic_functions.exp_decay import ExpDecay
import argparse
from robo.initial_design.init_random_uniform import init_random_uniform
from sklearn import preprocessing
from scipy.interpolate import spline
import matplotlib.pyplot as pl
from matplotlib.ticker import MaxNLocator
from matplotlib import cm


def plot_running(runCurves, predCurve, predStd2, trueCurve, conf=False):
	figure = pl.figure()
	predsteps = np.arange(len(predCurve))
	pl.plot(predsteps, predCurve, c='r', label='predicted')
	if conf:
		pl.fill(np.concatenate([predsteps[:-1], predsteps[:-1][::-1]]),
			np.concatenate([predCurve[:-1] - predStd2[:-1],
				(predCurve[:-1] + predStd2[:-1])[::-1]]),
			alpha=.5, fc='r', ec='None', label='confidence interval')
			
	pl.plot(np.arange(len(trueCurve)), trueCurve, c='k', label='ground truth')
	colors = cm.rainbow(np.linspace(0, 1, runCurves.shape[0] + 1))
	for i in xrange(len(runCurves)):
		yn = runCurves[i]
		tn = np.arange(len(yn))
		label = 'config ' + str(i+1)
		pl.plot(tn, yn, c=colors[i], marker='x')#, label=label
	
	pl.xlabel('steps')
	pl.ylabel('y values')
	gc = figure.gca()
	gc.xaxis.set_major_locator(MaxNLocator(integer=True))
	pl.legend(loc='upper right')
	pl.show()

def plot_mean_curve(means, std2s, curves, colors):
	plot_mean(means, std2s, colors)
	plot_curve(curves, colors)

def plot_mean(means, std2s, colors):
	figure = pl.figure()
	x = np.arange(1, len(means)+1) #configurations
	container = pl.errorbar(x, means, yerr=std2s, fmt='o')#, ecolor=colors)#, ecolor=colors)#,linestyle='None', marker='^')
	#xnew = np.linspace(x.min(), x.max(), 300)
	#power_smooth = spline(x, means, xnew)
	#pl.plot(xnew, power_smooth)
	pl.scatter(x, means, s=50, c=colors)
	connector, caplines, (vertical_lines,) = container.lines
	vertical_lines.set_color(colors)
	pl.ylabel('predicted asymptotic mean')
	pl.xlabel('configuration number')
	pl.show()

def plot_curve(curves, colors):
	figure = pl.figure()
	for i in xrange(len(curves)):
		yn = curves[i]
		tn = np.arange(len(yn))
		label = 'config ' + str(i+1)
		pl.plot(tn, yn, c=colors[i], label=label)

	ax = figure.gca()
	ax.xaxis.set_major_locator(MaxNLocator(integer=True))
	#pl.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3, fancybox=True, shadow=True)
	pl.xlabel('steps')
	pl.ylabel('f value')
	pl.show()

def example_predict_running():
	task = ExpDecay()
	#task.X_lower = np.array([70, 1])
	#task.X_upper = np.array([100, 5])
	task.X_lower = np.array([0., 0.])
	task.X_upper = np.array([1., 1.])
	#init = init_random_uniform(task.X_lower, task.X_upper, 5)
	init = init_random_uniform(task.X_lower, task.X_upper, 10)
	#control over the dimensions
	#min_max_scaler = preprocessing.MinMaxScaler()
	#init = min_max_scaler.fit_transform(init)

	tList = np.arange(3,21)

	ys = np.zeros(init.shape[0] +1, dtype=object)

	runnings = np.zeros(init.shape[0], dtype=object)

	for i in xrange(len(init)):
		nr_steps = np.random.choice(tList, 1)
		curve = task.f(t=np.arange(1,nr_steps[0]), x=init[i,:])
		curve = np.append(np.array([1.]), curve)
		#if i==0:
		#	curve = min_max_scaler.fit_transform(curve)
		#else:
		#	curve = min_max_scaler.transform(curve)
		print 'curve: ', curve
		ys[i] = curve
		runnings[i] = curve


	x_test = init_random_uniform(task.X_lower, task.X_upper, 1)
	#x_test = min_max_scaler.fit_transform(x_test)
	print 'x_test: ', x_test
	x_test2 = np.array([[x_test[0,0]*1.5, x_test[0,1]*1.5]])
	print 'x_test2: ', x_test2

	curve_complete = task.f(t=np.arange(1,21), x=x_test.flatten())
	curve_complete = np.append(np.array([1.]), curve_complete)
	#curve_complete = min_max_scaler.transform(curve_complete)

	curve_complete2 = task.f(t=np.arange(1,21), x=x_test2.flatten())
	curve_complete2 = np.append(np.array([1.]), curve_complete2)
	curve_partial2 = curve_complete2[:6]
	mu2 = curve_complete2[6:]


	curve_partial = curve_complete[:6]

	ys[-1] = curve_partial

	init = np.append(init, x_test, axis=0)

	freeze_thaw_model = FreezeThawGP(x_train=init, y_train=ys, invChol=True, lg=True)


	freeze_thaw_model.train()

	#_, _ = freeze_thaw_model.predict(option='asympt', xprime=x_test)
	_, _ = freeze_thaw_model.predict(option='asympt', xprime=init)

	mu, std2 = freeze_thaw_model.predict(option='old', conf_nr=init.shape[0]-1, from_step=7, further_steps=14)
	#control over dimensions
	pred_complete2 = np.append(curve_partial, mu2)
	#mu = min_max_scaler.transform(mu)

	pred_complete = np.append(curve_partial, mu)

	#plot_running(runCurves=runnings, predCurve=pred_complete, predStd2=std2, trueCurve=curve_complete, conf=False)
	plot_running(runCurves=runnings, predCurve=pred_complete2, predStd2=std2, trueCurve=curve_complete, conf=False)


def example_predict_asymptotic():
	color = ['b','g','c','m','y','k','r','g','c','m']
	grid = 5
	sigma = 0.1
	mu=0.
	task = ExpDecay()
	stepL=3
	stepU = 8
	stepR = np.arange(stepL, stepU+1)
	#task.X_lower = np.array([70, 1])
	#task.X_upper = np.array([100, 5])
	task.X_lower = np.array([0., 0.])
	task.X_upper = np.array([1., 1.])
	init = init_random_uniform(task.X_lower, task.X_upper, 10)
	ys = np.zeros(init.shape[0], dtype=object)
	for i in xrange(len(init)):
		steps = np.random.choice(stepR, 1)
		xs = np.linspace(1, steps, grid*steps)
		epslon = sigma*np.random.randn() + mu
		curve = task.f(t=np.arange(1, steps+1), x=init[i,:]) + epslon
		#print 'curve: ', curve
		ys[i] = curve

	freeze_thaw_model = FreezeThawGP(x_train=init, y_train=ys, invChol=True, lg=False)


	freeze_thaw_model.train()

	mu, std2 = freeze_thaw_model.predict(option='asympt', xprime=init)
	print 'std2: ', std2

	plot_mean_curve(means=np.abs(mu), std2s=std2, curves=ys, colors=color)

#fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True)
#ax = axs[0,0]

def main(predOld, predAsympt):
	if predOld:
		example_predict_running()
	if predAsympt:
		example_predict_asymptotic()


if __name__== "__main__":
	#example_predict_running()
	example_predict_asymptotic()
	# parser = argparse.ArgumentParser()
	# parser.add_argument('-o', action='store_true', default=False,
 #                    dest='predOld',
 #                    help='Testing prediction of further steps of old configurations')
	# parser.add_argument('-a', action='store_true', default=False,
 #                    dest='predAsympt',
 #                    help='Testing prediction of asymptotic values of completely new configurations')
	# arg = parser.parse_args()
	# main(arg.predOld, arg.predAsympt)
