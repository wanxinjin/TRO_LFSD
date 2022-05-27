import numpy as np
from CPDP import CPDP
from JinEnv import JinEnv
import casadi

import matplotlib.pyplot as plt

# set the plotting parameters
params = {'axes.labelsize': 25,
          'axes.titlesize': 25,
          'xtick.labelsize': 20,
          'ytick.labelsize': 20,
          'legend.fontsize': 20}
plt.rcParams.update(params)

# timing for different number of parameter (T=0.1 is fixed, and n_grid =2)
if True:
    p_dim = [10, 15, 20, 25, 40, 50, 75, 100]
    numeric_time = [0.37, 0.59, 0.74, 1.00, 1.9, 2.4, 3.64, 6.02]
    analyic_time = [0.07, 0.10, 0.11, 0.15, 0.17, 0.21, 0.33, 0.68]

    fig = plt.figure(2, figsize=(6, 4.5))
    ax = fig.subplots(1, 1)

    ax.set_facecolor('#E6E6E6')
    X = ['10', '15', '20', '25', '40', '50', '75', '100']
    X_axis = np.arange(len(X))
    ax.bar(X_axis - 0.2, numeric_time, 0.4, label='Numerical gradient')
    ax.bar(X_axis + 0.2, analyic_time, 0.4, label='Analytic gradient')
    ax.set_xticks(X_axis)
    ax.set_xticklabels(X)
    ax.set_xlabel(r'Dimension of $\theta$')
    ax.set_ylabel('Running time [sec]')
    ax.legend()
    ax.grid()
    ax.set_position([0.12, 0.16, 0.85, 0.81])

    plt.show()

# timing for different horizon length （theta's dim =40)
if True:
    horizon = [0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6]
    numeric_time = [2.000, 2.05, 2.577, 2.218, 2.1, 2.49, 2.310, 2.7]
    analyic_time = [0.503, 0.507, 0.690, 0.6811, 0.72, 0.718, 0.8298, 0.924]

    fig = plt.figure(2, figsize=(6, 4.5))
    ax = fig.subplots(1, 1)

    ax.set_facecolor('#E6E6E6')
    X = ['0.2', '0.4', '0.6', '0.8', '1.0', '1.2', '1.4', '1.6']
    X_axis = np.arange(len(X))
    ax.bar(X_axis - 0.2, numeric_time, 0.4, label='Numerical gradient')
    ax.bar(X_axis + 0.2, analyic_time, 0.4, label='Analytic gradient')
    ax.set_xticks(X_axis)
    ax.set_xticklabels(X)
    ax.set_xlabel(r'Horizon T')
    ax.set_ylabel('Running time [sec]')
    ax.set_ylim([0, 4])
    ax.legend()
    ax.grid()
    ax.set_position([0.15, 0.16, 0.83, 0.81])

    plt.show()
