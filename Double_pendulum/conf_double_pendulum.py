# -*- coding: utf-8 -*-
"""
Configuration file to Double_Pendulum
"""
import numpy as np
import matplotlib.pyplot as plt 

from time import time as clock
from time import sleep

time_start = clock()
SIMULATOR = "ideal" #switch btw "pinocchio" or "ideal"
DO_PLOTS = False

# General_Configuration
    # goal point
    
q_des = np.array([0, 0])

    # weight factor 
w_p = 1e2       # position weight
w_v = 1e-4      # velocity weight 
w_a = 1e-6      # acceleration weight
w_final = 1e2   # final cost weight --> not used

    # number of the initial states
number_init_state = 100

    # parameter for visualization 
np.set_printoptions(precision=2, linewidth=200, suppress=True)
LINE_WIDTH = 1
SIMULATOR = "ideal" #"mujoco" or "pinocchio" or "ideal"
nq = 2

    # simulation parameter  
simulate_coulomb_friction = 0    # flag specifying whether coulomb friction is simulated
simulation_type = 'euler' # either 'timestepping' or 'euler'
tau_coulomb_max = 0*np.ones(nq)   # expressed as percentage of torque max
randomize_robot_model = 0
use_viewer = True
which_viewer = 'meshcat'
simulate_real_time = 1          # flag specifying whether simulation should be real time or as fast as possible
show_floor = False
DISPLAY_T = 0.02              # update robot configuration in viwewer every DISPLAY_T seconds
    # constraint parameter
TAU_MAX = 1
TAU_MIN = -1
    # how to save csv file
booltrain_file = True # TRUE: ocp_dataset_DP_train  --- FALSE: ocp_dataset_DP_eval
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# OCP configuration
# ----------------------------------------------------------------------------------------------------

    # horizon parameters--> N
T = 0.20         # OCP horizon size
dt = 0.01        # OCP time step
N_step = int(T/dt);   # Number of horizon step
N_sim = 50


number_init_state = 11



# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
# MCP configuration
# ----------------------------------------------------------------------------------------------------

    # horizon parameters--> M
T = 0.20         # OCP horizon size
dt = 0.1        # OCP time step
M_step = int(T/dt);   # Number of horizon step
M_sim = 50

q0      = np.array([2,2]) 

# ----------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------
    # Configuration Neural Network
# ----------------------------------------------------------------------------------------------------

SCALE = 10000
