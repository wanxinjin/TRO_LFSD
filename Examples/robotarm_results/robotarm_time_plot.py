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

load1 = np.load('./robotarm_random.npy', allow_pickle=True).item()
loss_trace1 = load1['loss_trace']
parameter_trace1 = np.array(load1['parameter_trace'])
print(parameter_trace1[-1][0])
print(loss_trace1[-1])

load2 = np.load('./robotarm_time_2.npy', allow_pickle=True).item()
loss_trace2 = load2['loss_trace']
parameter_trace2 = np.array(load2['parameter_trace'])
print(parameter_trace2[-1][0:2])
print(loss_trace2[-1])



load3 = np.load('./robotarm_time_3.npy', allow_pickle=True).item()
loss_trace3 = load3['loss_trace']
parameter_trace3 = np.array(load3['parameter_trace'])
print(parameter_trace3[-1][0:3])
print(loss_trace3[-1])



load4 = np.load('./robotarm_time_4.npy', allow_pickle=True).item()
loss_trace4 = load4['loss_trace']
parameter_trace4 = np.array(load4['parameter_trace'])
print(parameter_trace4[-1][0:4])
print(loss_trace4[-1])
