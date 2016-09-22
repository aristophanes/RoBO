import numpy as np
from robo.models.freeze_thaw_model2 import FreezeThawGP
from robo.task.synthetic_functions.exp_decay import ExpDecay
from robo.initial_design.init_random_uniform import init_random_uniform
from scipy.interpolate import spline
import matplotlib.pyplot as pl
from matplotlib.ticker import MaxNLocator
from matplotlib import cm

#freeze_thaw_model = FreezeThawGP()

#task = ExpDecay()
"""
This version also with the predNew test
"""

def plot_running(runCurves, predCurve, predStd2, trueCurve):
	figure = pl.figure()
	predsteps = np.arange(len(predCurve))
	#pl.plot(predsteps[:-1], predCurve[:-1], c='r', label='predicted')
	pl.plot(predsteps, predCurve, c='r', label='predicted')
	# pl.fill(np.concatenate([predsteps[:-1], predsteps[:-1][::-1]]),
	# 		np.concatenate([predCurve[:-1] - predStd2[:-1],
	# 		(predCurve[:-1] + predStd2[:-1])[::-1]]),
	# 		alpha=.5, fc='r', ec='None', label='confidence interval')
			
	#pl.plot(np.arange(len(trueCurve)), trueCurve[::-1], c='k', label='ground truth')
	#pl.plot(np.arange(len(trueCurve))[:-1], trueCurve[:-1], c='k', label='ground truth')
	pl.plot(np.arange(len(trueCurve)), trueCurve, c='k', label='ground truth')
	colors=cm.rainbow(np.linspace(0, 1, runCurves.shape[0] + 1))
	for i in xrange(len(runCurves)):
		yn = runCurves[i]
		tn = np.arange(len(yn))
		label = 'config ' + str(i+1)
		#pl.plot(tn, yn[::-1], c=colors[i], label=label)
		#pl.plot(tn[:-1], yn[:-1], c=colors[i], marker='x', label=label)
		pl.plot(tn, yn, c=colors[i], marker='x')#, label=label
	
	pl.xlabel('steps')
	pl.ylabel('y values')
	ax = figure.gca()
	ax.xaxis.set_major_locator(MaxNLocator(integer=True))
	pl.legend(loc='upper right')
	pl.show()

def plot_mean_curve(means, std2s, curves, colors):
	plot_mean(means, std2s, colors)
	plot_curve(curves, colors)

def plot_mean(means, std2s, colors):
	figure = pl.figure()
	x = np.arange(1, len(means)+1) #configurations
	#pl.errorbar(x, means/100., yerr=np.sqrt(std2s)*0.02, fmt='-')
	#pl.ylim(0., 0.9)
	pl.scatter(x, means, s=50, c=colors)
	#pl.stem(x, means/10, markerfmt=' ')
	#power = np.array([1.53E+03, 5.92E+02, 2.04E+02, 7.24E+01, 2.72E+01, 1.10E+01, 4.70E+00, 3.20E+00, 5.10E+00, 4.30E+00]
	xnew = np.linspace(x.min(), x.max(), 300)
	#power_smooth = spline(x, means/10, xnew)
	power_smooth = spline(x, means, xnew)
	pl.plot(xnew, power_smooth)
	pl.ylabel('predicted\nasymptotic\nmean')
	pl.xlabel('configuration number')
	pl.show()
	#pl.show()

def plot_curve(curves, colors):
	figure = pl.figure()
	for i in xrange(len(curves)):
		yn = curves[i]
		tn = np.arange(len(yn))
		#tnew = np.linspace(tn.min(), tn.max(), 300)
		#power_smooth = spline(tn, yn, tnew)
		label = 'config ' + str(i+1)
		pl.plot(tn, yn, c=colors[i], label=label)
		#pl.plot(tnew, power_smooth, c=colors[i], label=label)

	ax = figure.gca()
	ax.xaxis.set_major_locator(MaxNLocator(integer=True))
	pl.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3, fancybox=True, shadow=True)
	pl.xlabel('steps')
	pl.ylabel('loss')
	pl.show()

def example_predict_running():
	task = ExpDecay()
	#init = init_random_uniform(task.X_lower, task.X_upper, 5)
	init = init_random_uniform(task.X_lower, task.X_upper, 10)
	#print init
	tList = np.arange(2,6)
	ys = np.zeros(init.shape[0] +1, dtype=object)
	runnings = np.zeros(init.shape[0], dtype=object)
	for i in xrange(len(init)):
		nr_steps = np.random.choice(tList, 1)
		curve = task.f(t=np.arange(1,nr_steps[0]), x=init[i,:])
		print 'curve: ', curve
		ys[i] = curve
		runnings[i] = curve


	x_test = init_random_uniform(task.X_lower, task.X_upper, 1)
	#print 'x_test: ', x_test
	curve_complete = task.f(t=np.arange(1,6), x=x_test.flatten())
	#print 'curve_test: ', curve_complete

	curve_partial = curve_complete[:2]

	ys[-1] = curve_partial
	#print
	#print 'ys: '
	#print ys
	#print
	init = np.append(init, x_test, axis=0)
	#print 'init: ', init

	freeze_thaw_model = FreezeThawGP(x_train=init, y_train=ys, invChol = True, lg=False)


	freeze_thaw_model.train()

	_, _ = freeze_thaw_model.predict(option='asympt', xprime=x_test)

	mu, std2 = freeze_thaw_model.predict(option='old', conf_nr=init.shape[0]-1, from_step=3, further_steps=3)

	#print
	#print 'pred: ', mu
	#print
	#print 'ground: ', curve_complete[2:]
	pred_complete = np.append(curve_partial, mu)

	plot_running(runCurves=runnings, predCurve=pred_complete, predStd2=std2, trueCurve=curve_complete)



# def example_predict_asymptotic():
# 	color = ['b','g','c','m','y','k','r','g','c','m']
# 	task = ExpDecay()
# 	#task.X_lower = np.array([70, 1])
# 	#task.X_upper = np.array([100, 5])
# 	init = init_random_uniform(task.X_lower, task.X_upper, 10)
# 	ys = np.zeros(init.shape[0], dtype=object)
# 	for i in xrange(len(init)):
# 		curve = task.f(t=np.arange(1,6), x=init[i,:])
# 		#print 'curve: ', curve
# 		ys[i] = curve

# 	freeze_thaw_model = FreezeThawGP(x_train=init, y_train=ys, invChol=True, lg=False)


# 	freeze_thaw_model.train()

# 	mu, std2 = freeze_thaw_model.predict(option='asympt', xprime=init)

# 	#print 'mu: ', mu
# 	#print 'std2: ', std2

# 	plot_mean_curve(means=mu, std2s=std2, curves=ys, colors=color)

def example_predict_asymptotic():
	color = ['b','g','c','m','y','k','r','g','c','m']
	grid = 5
	sigma = 0.1
	mu=0.
	task = ExpDecay()
	stepL=3
	stepU = 8
	stepR = np.arange(stepL, stepU+1)
	task.X_lower = np.array([70, 1])
	task.X_upper = np.array([100, 5])
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

	#print 'mu: ', mu
	#print 'std2: ', std2

	plot_mean_curve(means=np.abs(mu), std2s=std2, curves=ys, colors=color)

def example_predict_new(conf=False):
	task = ExpDecay()
	task.X_upper = np.array([0.02,0.02])
	#init = init_random_uniform(task.X_lower, task.X_upper, 10)
	init = init_random_uniform(task.X_lower, task.X_upper, 20)
	ys = np.zeros(init.shape[0], dtype=object)
	for i in xrange(len(init)):
		curve = task.f(t=np.arange(1,6), x=init[i,:])
		#print 'curve: ', curve
		ys[i] = curve

	freeze_thaw_model = FreezeThawGP(x_train=init, y_train=ys, invChol=True, lg=False)


	freeze_thaw_model.train()

	x_test = init_random_uniform(task.X_lower, task.X_upper, 1)

	x_test = init[0,:]

	y_test = task.f(t=np.arange(1,6), x=x_test.flatten())

	y_test = ys[0]

	_, _ = freeze_thaw_model.predict(option='asympt', xprime=x_test)

	mu, std2 = freeze_thaw_model.predict(xprime=x_test, option='new', further_steps=len(y_test))
	#std2 = std2[:,np.newaxis]
	mu = mu.flatten()
	#print 'mu: ', mu
	#print 'std2: ', std2
	pl.figure()
	pl.ylabel('Loss')
	pl.xlabel('Step')
	pl.grid(True)

	xs = np.arange(y_test.shape[0])
	pl.plot(xs, y_test, 'k-', label = "ground truth")
	pl.plot(xs, mu, 'b-', label = "predicted")
	if conf:
		pl.fill(np.concatenate([xs, xs[::-1]]), np.concatenate([mu - 1.96 * std2, (mu + 1.96 * std2)[::-1]]), alpha=.5, fc='b', ec='None', label='confidence')
	pl.legend(loc='upper left', shadow=True)
	pl.show()


#example_predict_running()
example_predict_asymptotic()
#example_predict_new()