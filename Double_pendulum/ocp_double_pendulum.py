from example_robot_data.robots_loader import load
from adam.casadi.computations import KinDynComputations
import numpy as np
import casadi as cs
import os
from utils.robot_simulator import RobotSimulator
from utils.robot_loaders import loadUR
from utils.robot_wrapper import RobotWrapper
import torch

import pandas as pd

from time import time as clock
from time import sleep

import matplotlib.pyplot as plt
from termcolor import colored

import conf_double_pendulum as config
from neural_network_doublependulum import NeuralNetwork

from matplotlib.animation import FuncAnimation
class DoublePendulumOCP:
    
    def __init__(self,filename, robot_model="double_pendulum"):
        
        self.robot  = load(robot_model)
        self.kinDyn = KinDynComputations(self.robot.urdf, [s for s in self.robot.model.names[1:]])
        self.nq     = len(self.robot.model.names[1:])
        
        self.number_init_state = config.n_init_state_ocp
        self.q1_list,self.v1_list, self.q2_list,self.v2_list = self.set_initial_state_list()
        
        self.N  =    config.N_step
        self.M  =    config.M_step
        
        self.q_des  = config.q_des
        self.nx     = 2 * self.nq   #2x2 for double pendulum
        
        self.w_p     =   config.w_p
        self.w_v     =   config.w_v
        self.w_a     =   config.w_a
        self.w_final =   config.w_final
        
        self.inv_dyn,self.dynamic_f     = self.set_dynamics()
        
        self.filename   = filename
        
        print("Initialization Double_Pendulum OCP complete!")

    def set_initial_state_list(self):
        q_min, q_max = 0.0, np.pi
        dq_min, dq_max = 0.0, 8.0

        q1_list = np.zeros((self.number_init_state, 1))
        v1_list = np.zeros((self.number_init_state, 1))
        q2_list = np.zeros((self.number_init_state, 1))
        v2_list = np.zeros((self.number_init_state, 1))

        if config.random_initial_set:
            # Genera tutto random uniformemente nello spazio [q_min,q_max], [dq_min,dq_max]
            q1_list = np.random.uniform(q_min,  q_max,  (self.number_init_state, 1))
            q2_list = np.random.uniform(q_min,  q_max,  (self.number_init_state, 1))
            v1_list = np.random.uniform(dq_min, dq_max, (self.number_init_state, 1))
            v2_list = np.random.uniform(dq_min, dq_max, (self.number_init_state, 1))

        else:
            # Genera una discretizzazione semplice lineare e applica uno shift!!!! 
            phi = np.pi / 4  
            dq_phi = 2.0

            q_lin  = np.linspace(q_min,  q_max,  self.number_init_state)
            dq_lin = np.linspace(dq_min, dq_max, self.number_init_state)

            for i in range(self.number_init_state):
                # q1, v1 come da griglia!!!
                q1_list[i, 0] = q_lin[i]
                v1_list[i, 0] = dq_lin[i]

                # q2 è q1 + phi, con eventuale "avvolgimento", zio pera !!!
                q2_val = q_lin[i] + phi
                if q2_val > q_max:
                    q2_val -= (q_max - q_min)
                q2_list[i, 0] = q2_val

                v2_val = dq_lin[i] + dq_phi
                if v2_val > dq_max:
                    v2_val -= (dq_max - dq_min)
                v2_list[i, 0] = v2_val

        return q1_list, v1_list, q2_list, v2_list
    
    def set_dynamics(self):
        #use the alternative multi-body dynamics modeling 
        q       = cs.SX.sym("q", self.nq)   #position
        dq      = cs.SX.sym("dq", self.nq)  #velocity
        ddq     = cs.SX.sym("ddq", self.nq) #acceleration is our control input --> because we use the inverse dynamic problem

        state   = cs.vertcat(q, dq)     #vertical concatenation q and dq
        rhs     = cs.vertcat(dq, ddq)   #vertical concatenation dq and ddq
        
        dynamic_f  = cs.Function('f', [state, ddq], [rhs]) #dynamic function, take in input the state and ddq"=u" (x, u = ddq) and compute dx =(dq,ddq)
        #inverse dynamic function with casadi
        H_b     = cs.SX.eye(4)
        v_b     = cs.SX.zeros(6)
        M = self.kinDyn.mass_matrix_fun()(H_b, q)[6:, 6:]      # excluding the first 6 elements 
        #(which usually represent the degrees of freedom of the basic rigid body, such as translations and rotations if the robot is floating).
        h = self.kinDyn.bias_force_fun()(H_b, q, v_b, dq)[6:]  # excluding the base DoF
        tau = M @ ddq + h           # out control input tau = M(q)*ddq + h(q,dq)

        inv_dyn = cs.Function('inv_dyn', [state, ddq], [tau])
        return inv_dyn,dynamic_f
   
    def setup_ocp(self, q0, dq0, with_N=True, with_M=False):
        self.opti = cs.Opti()
        
        if with_N and with_M:
            iteration = self.N + self.M
        elif with_N:
            iteration = self.N
        elif with_M:
            iteration = self.M
        else:
            iteration = self.N
            
        # ottimization variable
        X, U = [], []
        for i in range(iteration + 1):
            X.append(self.opti.variable(self.nx))
        for i in range(iteration):
            U.append(self.opti.variable(self.nq))
        
        # parameters (not variables!!!): stato iniziale e q_des
        param_x_init = self.opti.parameter(self.nx)
        param_q_des  = self.opti.parameter(self.nq)
        
        # Setting of the value of the parameters
        x_init = np.concatenate([q0, dq0])
        self.opti.set_value(param_x_init, x_init)
        self.opti.set_value(param_q_des,  self.q_des)
        
        # constraint to the initial state
        self.opti.subject_to(X[0] == param_x_init)
        
        cost_expr = 0
        
        for i in range(iteration):
            # error of the position and velocity
            q_error  = X[i][:self.nq] - param_q_des
            dq_error = X[i][self.nq:]
            
            cost_expr += self.w_p * cs.dot(q_error,  q_error)
            cost_expr += self.w_v * cs.dot(dq_error, dq_error)
            cost_expr += self.w_a * cs.dot(U[i], U[i])  
            
            # dynamic implementation
            x_next = X[i] + config.dt * self.dynamic_f(X[i], U[i])
            self.opti.subject_to(X[i+1] == x_next)

            # "tourque" limit
            tau = self.inv_dyn(X[i], U[i])
            self.opti.subject_to(self.opti.bounded(config.TAU_MIN, tau, config.TAU_MAX))
        
        # add terminal cost and terminal constraint
        # q_error_final  = X[-1][:self.nq] - param_q_des
        # dq_error_final = X[-1][self.nq:]
        # cost_expr     += self.w_final * cs.dot(q_error_final,  q_error_final)
        # cost_expr     += self.w_final * cs.dot(dq_error_final, dq_error_final)
        
        # dq_final = X[iteration][self.nq:]
        # self.opti.subject_to(dq_final == 0.0)
        
        self.opti.minimize(cost_expr)
        
        # Solver
        opts = {
            "ipopt.print_level":    0,
            "ipopt.max_iter":       config.max_iter_opts,
            "print_time":           0,
            "ipopt.tol":            1e-3,
            "ipopt.constr_viol_tol":1e-6
        }
        self.opti.solver("ipopt", opts)
        sol = self.opti.solve()
        
        # Recover of the date trajectory and optimal cost
        x_sol   = np.array([sol.value(X[k]) for k in range(iteration+1)]).T
        u_sol   = np.array([sol.value(U[k]) for k in range(iteration)]).T
        q_traj  = x_sol[:self.nq, :].T
        dq_traj = x_sol[self.nq:, :].T
        final_cost = self.opti.value(cost_expr)
        
        return sol, final_cost, x_sol, u_sol, q_traj, dq_traj
    
def simulation(ocp_double_pendulum,with_N = True, with_M= False):
    number_init_state=config.n_init_state_ocp
    state_buffer = []       # Buffer to store initial states
    cost_buffer = []        # Buffer to store optimal costs
    #simulation for each type of the initial state
    for current_state in range(number_init_state):
        q0 = np.array([ocp_double_pendulum.q1_list[current_state][0], ocp_double_pendulum.q2_list[current_state][0]])
        dq0 = np.array([ocp_double_pendulum.v1_list[current_state][0], ocp_double_pendulum.v2_list[current_state][0]])

        try:
            q0 = np.array([ocp_double_pendulum.q1_list[current_state][0], ocp_double_pendulum.q2_list[current_state][0]])
            dq0 = np.array([ocp_double_pendulum.v1_list[current_state][0], ocp_double_pendulum.v2_list[current_state][0]])
            print("____________________________________________________________")
            print(f"Start computation OCP... Configuration {current_state+1}:")
            sol, final_cost, x_sol, u_sol, q_trajectory, dq_trajectory = ocp_double_pendulum.setup_ocp(q0, dq0, with_N, with_M)
            print(f"        Initial position (q0): {q0}")
            print(f"        Initial velocity (dq0): {dq0}")
            print(f"        Desired postiion (q):  ",ocp_double_pendulum.q_des)
            print(f"        Final_cost {current_state+1}: ",final_cost)
            
            state_buffer.append ([ocp_double_pendulum.q1_list[current_state][0], ocp_double_pendulum.q2_list[current_state][0],ocp_double_pendulum.v1_list[current_state][0], ocp_double_pendulum.v2_list[current_state][0]])
            cost_buffer.append(final_cost)
            
        except RuntimeError as e:
            if "Infeasible_Problem_Detected" in str(e):
                print(f"Could not solve for: ")
                print(f"        Initial position (q0): {q0}")
                print(f"        Initial velocity (dq0): {dq0}")
            else:
                print("Runtime error:", e)
        print ("____________________________________________________________")         
    return state_buffer,cost_buffer
    
def save_result(filename, state_buffer, cost_buffer):
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        positions_q1  = [state[0] for state in state_buffer]
        positions_q2  = [state[1] for state in state_buffer]
        velocities_v1 = [state[2] for state in state_buffer]
        velocities_v2 = [state[3] for state in state_buffer]
        
        df = pd.DataFrame({'q1': positions_q1, 'q2': positions_q2, 'v1': velocities_v1, 'v2': velocities_v2, 'cost': cost_buffer})
        df.to_csv(filename, index=False)
        
        print(f"File saved: {filename}")

# ---------------------------------------------------------------------
#          MAIN(example)
# ---------------------------------------------------------------------
    
if __name__ == "__main__":
    time_start = clock()
    with_N = True
    with_M = False
    save_result_bool = True
    train_nn    = True
    
    print("START THE PROGRAM:")
    print(f"Setup choice:number initial states: {config.n_init_state_ocp}, N={config.N_step}, M={config.M_step}, tau_min and max={config.TAU_MAX}, max_iter={config.max_iter_opts}")
    print(f"boolean value: with_N={with_N}, with_M={with_M}, save_result={save_result_bool}, train_nn={train_nn}, Random distribution {config.random_initial_set}")
    print("press a button to continue")
    input()
    filename = config.csv_eval
    
    ocp_double_pendulum = DoublePendulumOCP(filename)
    state_buffer,cost_buffer = simulation(ocp_double_pendulum,with_N,with_M)
    
    if (save_result_bool):
        save_result(filename,state_buffer,cost_buffer)
        print("finish save result")
        
    if (train_nn):   
        nn = NeuralNetwork(filename,ocp_double_pendulum.nx)
        nn.trainig_part()
        nn.plot_training_history()
        # Save the trained model
        os.makedirs("models", exist_ok=True)
        torch.save( {'model':nn.state_dict()}, "models/model.pt")
    
    print("Total script time:", clock() - time_start)











