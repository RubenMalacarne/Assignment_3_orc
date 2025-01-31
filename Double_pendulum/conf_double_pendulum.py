# -*- coding: utf-8 -*-
"""
Configuration file for Single Pendulum
"""
import numpy as np
import matplotlib.pyplot as plt
from time import time as clock, sleep

# GOAL POINT and INITIAL CONFIGURATION  ----------------------------------------------------------
q_des = np.array([0,0])     #"pendolo a piombo" --> np.array([-np.pi, -np.pi*2])
q0 = np.array([np.pi, np.pi])   #serve solo come primo punto per il simulatore 3D poi viene reimpostato
random_initial_set = True
n_init_state_ocp = 1100 #100 for eval  # Numbers of initial value

# STEP VALUE  ------------------------------------------------------------------------------------
dt_sim = 0.002 
dt = 0.01       # OCP time step
N_sim = 500     # Simulation steps for MPC

N_step = 100     # Time horizon N steps for OCP
M_step = 6     # Time Horizon M steps for OCP

max_iter_opts = 1000
SOLVER_MAX_ITER = 100

# LENGTH LINK 1 and LINK 2 ------------------------------------------------------------------------
L1 = 0.1  
L2 = 0.2    
# Constraint add, in URDF file are not present  ---------------------------------------------------
value_tau = 1
TAU_MAX = value_tau
TAU_MIN = -(value_tau)

# WEIGHT FACTOR  ----------------------------------------------------------------------------------
w_p     = 1e1   # position weight
w_v     = 1e-3  # velocity weight
w_a     = 1e-6  # Acceleration weight
w_final = 1e2   # Final cost weight (not used)
w_value_nn = 1e-1
# w_p     = 1e1   # position weight
# w_v     = 0e-6   # velocity weight
# w_a     = 1e-5  # Acceleration weight
# w_final = 1e0   # Final cost weight (not used)


# FILE PATH ----------------------------------------------------------------------------------------
csv_train = 'dataset/ocp_dataset_DP_train.csv'
csv_eval ='dataset/ocp_dataset_DP_eval.csv'

# SIMULATOR SETTINGS -------------------------------------------------------------------------------
np.set_printoptions(precision=2, linewidth=200, suppress=True)
LINE_WIDTH = 1
SIMULATOR = "pinocchio"#"ideal"  # Options: "pinocchio", "ideal"
DO_PLOTS = False
nq = 2

simulate_coulomb_friction = 0  # Flag for simulating Coulomb friction
simulation_type = 'euler'      # Options: 'timestepping', 'euler'
tau_coulomb_max = 0 * np.ones(nq)  # Max Coulomb torque as percentage
randomize_robot_model = 0
use_viewer = True
which_viewer = 'meshcat'
simulate_real_time = 1         # Real-time simulation flag
show_floor = False
DISPLAY_T = 0.02               # Viewer update interval (seconds)


