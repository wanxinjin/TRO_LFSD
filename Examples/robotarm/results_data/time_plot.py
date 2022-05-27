import matplotlib.pyplot as plt
import numpy as np

# set the plotting parameters
params = {'axes.labelsize': 20,
          'axes.titlesize': 20,
          'xtick.labelsize': 15,
          'ytick.labelsize': 15,
          'legend.fontsize': 16}
plt.rcParams.update(params)

# load date
np.set_printoptions(precision=3)

# load the data
trials = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

print('\n--------------------------------------------------')

loss_all_trials = []
learned_param_all_trials = []
for k in trials:
    load = np.load('time/tworder_' + str(1) + '_trial_' + str(k) + '.npy', allow_pickle=True).item()
    loss_all_trials.append(load['loss_trace'])
    learned_param_all_trials.append(load['parameter_trace'][-1])
loss_all_trials = np.array(loss_all_trials)
learned_param_all_trials = np.array(learned_param_all_trials)
loss_mean = loss_all_trials.mean(axis=0)
loss_std = loss_all_trials.std(axis=0)
learned_param_mean = learned_param_all_trials.mean(axis=0)
learned_param_std = learned_param_all_trials.std(axis=0)
print('tw_order:', 1)
print('loss mean', loss_mean[-1], )
print('loss std', loss_std[-1])
print('tw param mean', learned_param_mean[0])
print('tw param std', learned_param_std[0])

print('\n--------------------------------------------------')
loss_all_trials = []
learned_param_all_trials = []
for k in trials:
    load = np.load('time/tworder_' + str(2) + '_trial_' + str(k) + '.npy', allow_pickle=True).item()
    loss_all_trials.append(load['loss_trace'])
    learned_param_all_trials.append(load['parameter_trace'][-1])
loss_all_trials = np.array(loss_all_trials)
learned_param_all_trials = np.array(learned_param_all_trials)
loss_mean = loss_all_trials.mean(axis=0)
loss_std = loss_all_trials.std(axis=0)
learned_param_mean = learned_param_all_trials.mean(axis=0)
learned_param_std = learned_param_all_trials.std(axis=0)
print('tw_order:', 2)
print('loss mean', loss_mean[-1], )
print('loss std', loss_std[-1])
print('tw param mean', learned_param_mean[0:2])
print('tw param std', learned_param_std[0:2])

print('\n--------------------------------------------------')

loss_all_trials = []
learned_param_all_trials = []
for k in trials:
    load = np.load('time/tworder_' + str(3) + '_trial_' + str(k) + '.npy', allow_pickle=True).item()
    loss_all_trials.append(load['loss_trace'])
    learned_param_all_trials.append(load['parameter_trace'][-1])
loss_all_trials = np.array(loss_all_trials)
learned_param_all_trials = np.array(learned_param_all_trials)
loss_mean = loss_all_trials.mean(axis=0)
loss_std = loss_all_trials.std(axis=0)
learned_param_mean = learned_param_all_trials.mean(axis=0)
learned_param_std = learned_param_all_trials.std(axis=0)
print('tw_order:', 3)
print('loss mean', loss_mean[-1], )
print('loss std', loss_std[-1])
print('tw param mean', learned_param_mean[0:3])
print('tw param std', learned_param_std[0:3])




print('\n--------------------------------------------------')
trials = [0, 1, 2, 3, 4, 5, 6, ]
loss_all_trials = []
learned_param_all_trials = []
for k in trials:
    load = np.load('time/tworder_' + str(4) + '_trial_' + str(k) + '.npy', allow_pickle=True).item()
    loss_all_trials.append(load['loss_trace'])
    learned_param_all_trials.append(load['parameter_trace'][-1])
loss_all_trials = np.array(loss_all_trials)
learned_param_all_trials = np.array(learned_param_all_trials)
loss_mean = loss_all_trials.mean(axis=0)
loss_std = loss_all_trials.std(axis=0)
learned_param_mean = learned_param_all_trials.mean(axis=0)
learned_param_std = learned_param_all_trials.std(axis=0)
print('tw_order:', 4)
print('loss mean', loss_mean[-1], )
print('loss std', loss_std[-1])
print('tw param mean', learned_param_mean[0:4])
print('tw param std', learned_param_std[0:4])
