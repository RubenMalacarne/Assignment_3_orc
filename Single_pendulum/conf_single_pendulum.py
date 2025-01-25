# -*- coding: utf-8 -*-
"""
Configuration file for Double Pendulum
"""
import numpy as np
import matplotlib.pyplot as plt
from time import time as clock, sleep

# Goal point
q_des = np.array([0])# Default: "pendolo a piombo" --> np.array([-np.pi, -np.pi*2])

# Horizon parameters
dt = 0.01       # OCP time step
N_sim = 200     # Simulation steps for MPC
N_step = 40     # Time horizon N steps for OCP
M_step = 5     # Time Horizon M steps for OCP
random_initial_set = True
max_iter_opts = 1000
SOLVER_MAX_ITER = 100
# Initial configuration
q0 = np.array([np.pi])
# length of link 1 and 2
L1 = 0.035  
# Constraint parameters # These limits are not present in the URDF file
value_tau = 0.6
TAU_MAX = value_tau
TAU_MIN = -(value_tau)

# Initial states
n_init_state_ocp = 1100 # Avoid odd numbers if disliked

# Weight factors
w_p     = 0.1   # position weight
w_v     = 1e-4  # velocity weight
w_a     = 1e-6  # Acceleration weight
w_final = 1e3   # Final cost weight (not used)
w_value_nn = 1e1
# Visualization parameters  


#file
#train csv 
csv_train = 'dataset/ocp_dataset_SP_train.csv'
csv_eval ='dataset/ocp_dataset_SP_eval.csv'


# Simulator settings
np.set_printoptions(precision=2, linewidth=200, suppress=True)
LINE_WIDTH = 1
SIMULATOR = "ideal"  # Options: "pinocchio", "ideal"
DO_PLOTS = False
nq = 1

# Simulation parameters
simulate_coulomb_friction = 0  # Flag for simulating Coulomb friction
simulation_type = 'euler'      # Options: 'timestepping', 'euler'
tau_coulomb_max = 0 * np.ones(nq)  # Max Coulomb torque as percentage
randomize_robot_model = 0
use_viewer = True
which_viewer = 'meshcat'
simulate_real_time = 1         # Real-time simulation flag
show_floor = False
DISPLAY_T = 0.02               # Viewer update interval (seconds)


