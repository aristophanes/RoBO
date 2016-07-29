import numpy as np
from robo.initial_design.init_random_uniform import init_random_uniform
from robo.task.base_task import BaseTask
from robo.models.freeze_thaw_model1 import FreezeThawGP
from robo.maximizers.direct import Direct
from robo.acquisition.ei import EI
from robo.acquisition.information_gain_mc_freeze import InformationGainMC
from scipy.stats import norm
from robo.solver.freeze_thaw_bayesian_optimization import FreezeThawBO
from robo.task.synthetic_functions.exp_decay import ExpDecay



freeze_thaw_model = FreezeThawGP()

task = ExpDecay()

acquisition_func = EI(freeze_thaw_model, X_upper=np.array([0.1,0.1]), X_lower=np.array([0.01,0.01]))

maximizer = Direct(acquisition_func, np.array([0.01,0.01]), np.array([0.1,0.1]))

bo = FreezeThawBO(acquisition_func=acquisition_func,
                          freeze_thaw_model=freeze_thaw_model,
                          maximize_func=maximizer,
                          task=task)

incumbent, incumbent_value = bo.run(6)
print 'incumbent: ', incumbent
print 'incumbent_value: ', incumbent_value