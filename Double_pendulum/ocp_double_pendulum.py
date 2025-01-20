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
        self.nx     = 2 * self.nq
        
        self.w_p     =   config.w_p
        self.w_v     =   config.w_v
        self.w_a     =   config.w_a
        self.w_final =   config.w_final
        
        self.inv_dyn,self.dynamic_f = self.set_dynamics()
        
        self.filename= filename
        
        print("Initializeation Double_Pendulum OCP complete!")

    def set_initial_state_list(self):
        
        n_qs = self.number_init_state
        n_dqs = self.number_init_state
        q_min = 0
        q_max = np.pi
        dq_min = 0.0
        dq_max = 8.0

        # Sfasamento
        phi = np.pi / 4  
        dq_phi = 2.0

        q_step = (q_max - q_min) / (n_qs - 1)
        dq_step = (dq_max - dq_min) / (n_dqs - 1)
        qs = np.arange(q_min, q_max + q_step, q_step).reshape(n_qs, 1)
        dqs = np.arange(dq_min, dq_max + dq_step, dq_step).reshape(n_dqs, 1)

        # Velocità angolari
        dqs2 = (dqs + dq_phi) % (dq_max - dq_min) + dq_min
        return qs, dqs, qs + phi, dqs2
    
    def set_dynamics(self):
        #use the alternative multi-body dynamics modeling 
        q       = cs.SX.sym("q", self.nq)
        dq      = cs.SX.sym("dq", self.nq)
        ddq     = cs.SX.sym("ddq", self.nq) #is our control input (acceleration) --> becaus use the inverse dynamic problem

        state   = cs.vertcat(q, dq)     #vertical concatenation q and dq
        rhs     = cs.vertcat(dq, ddq)   #vertical concatenation dq and ddq
        
        dynamic_f  = cs.Function('f', [state, ddq], [rhs]) #dynamic function, take in input the state and ddq"=u" (x, u = ddq) and compute dx =(dq,ddq)
        #inverse dynamic function with casadi
        H_b     = cs.SX.eye(4)
        v_b     = cs.SX.zeros(6)
        M = self.kinDyn.mass_matrix_fun()(H_b, q)[6:, 6:]        # excluding the first 6 elements (which usually represent the degrees of freedom of the basic rigid body, such as translations and rotations if the robot is floating).
        h = self.kinDyn.bias_force_fun()(H_b, q, v_b, dq)[6:]   #excluding the base DoF
        tau = M @ ddq + h       # out control input tau = M(q)*ddq + h(q,dq)

        inv_dyn = cs.Function('inv_dyn', [state, ddq], [tau])
    
        return inv_dyn,dynamic_f
   
    def setup_ocp(self, q0, dq0, q_des, with_N = True, with_M= False):
        self.opti = cs.Opti()
        #set iteretion
        iteration = 0
        if (with_N) : iteration = self.N
        elif (with_M) : iteration = self.M
        elif (with_N and with_M): iteration = self.N + self.M 
        #inizialization
        self.cost = 0.0
        self.running_cost = [0]*(iteration)
        
        param_x_init = self.opti.parameter(self.nx)
        param_q_des  = self.opti.parameter(self.nq)
        
        x = np.concatenate([q0, dq0])
        
        self.opti.set_value(param_x_init, x)
        self.opti.set_value(param_q_des, q_des)
        
        #add dynamics
        inv_dyn = self.inv_dyn
        dynamic_f = self.dynamic_f
        
        X, U = [], []
        for i in range(iteration + 1):
            X.append(self.opti.variable(self.nx))
        for i in range(iteration):
            U.append(self.opti.variable(self.nq))
        
        self.opti.subject_to(X[0] == param_x_init)
        for i in range(iteration):
            self.running_cost[i] += self.w_p * X[i][:self.nq].T @ X[i][:self.nq]  # Position cost
            self.running_cost[i] += self.w_v * X[i][self.nq:].T @ X[i][self.nq:]  # Velocity cost
            self.running_cost[i] += self.w_a * U[i].T @ U[i]                      # Acceleration cost
            
            x_next = X[i] + config.dt * dynamic_f(X[i], U[i]) #computation of the next state
            self.opti.subject_to(X[i + 1] == x_next)

            tau = inv_dyn(X[i], U[i])
            self.opti.subject_to(self.opti.bounded(config.TAU_MIN, tau, config.TAU_MAX))
            #not add other inequelity constraint!
            self.cost += self.running_cost[i]
        
        self.opti.minimize(self.cost)
        
        opts = {
            "ipopt.print_level": 0,
            "ipopt.hessian_approximation": "limited-memory",
            "print_time": 0, # print information about execution time
            "detect_simple_bounds": True,
            "ipopt.max_iter": config.max_iter_opts #default 3000
        }
        # opts = {
        #     "ipopt.print_level": 0,
        #     "ipopt.tol": 1e-6,
        #     "ipopt.constr_viol_tol": 1e-6,
        #     "ipopt.compl_inf_tol": 1e-6,
        #     "print_time": 0,
        #     "detect_simple_bounds": True
        # }
        
        self.opti.solver("ipopt", opts)
        sol = self.opti.solve()
        q_trajectory = np.array([sol.value(X[i][:self.nq]) for i in range(iteration + 1)])
        x_sol = np.array([sol.value(X[k]) for k in range(iteration+1)]).T
        u_sol = np.array([sol.value(U[k]) for k in range(iteration)]).T
        return sol,q_trajectory,x_sol,u_sol
    #   final_cost  = self.opti.value(cost)
    #     #recover the solution
    #     x_sol = np.array([sol.value(X[k]) for k in range(iteration+1)]).T
    #     u_sol = np.array([sol.value(U[k]) for k in range(iteration)]).T
    #     q_trajectory = np.array([sol.value(X[i][:self.nq]) for i in range(iteration + 1)])
        
    #     q_sol = x_sol[:self.nq,:]
    #     dq_sol = x_sol[self.nq:,:]
        
    #     #compute the tau for each value
    #     tau_value = np.zeros((self.nq, iteration))
    #     for i in range(iteration):
    #         tau_value[:,i] = inv_dyn(x_sol[:,i], u_sol[:,i]).toarray().squeeze()
            
    #     final_q =  q_sol[:,iteration]
    #     final_dq= dq_sol[:,iteration]
def simulation(ocp_double_pendulum,with_N = True, with_M= False):
    number_init_state=config.n_init_state_ocp
    state_buffer = []       # Buffer to store initial states
    cost_buffer = []        # Buffer to store optimal costs
    #simulation for each type of the initial state
    for current_state in range(number_init_state):
        #initial state for q1,q2,v1,v2
        q0 = np.array([ocp_double_pendulum.q1_list[current_state][0], ocp_double_pendulum.q2_list[current_state][0]])
        dq0 = np.array([ocp_double_pendulum.v1_list[current_state][0], ocp_double_pendulum.v2_list[current_state][0]])

        try:
            q0 = np.array([ocp_double_pendulum.q1_list[current_state][0], ocp_double_pendulum.q2_list[current_state][0]])
            dq0 = np.array([ocp_double_pendulum.v1_list[current_state][0], ocp_double_pendulum.v2_list[current_state][0]])
            print("____________________________________________________________")
            print(f"Start computation OCP... Configuration {current_state+1}:")
            sol,q_trajectory,x_sol,u_sol= ocp_double_pendulum.setup_ocp(q0, dq0, ocp_double_pendulum.q_des,with_N,with_M)
            print(f"        Initial position (q0): {q0}")
            print(f"        Initial velocity (dq0): {dq0}")
            print(f"        Desired postiion (q):  ",ocp_double_pendulum.q_des)
            print(f"        Final_cost {current_state+1}: ",sol.value(ocp_double_pendulum.cost))
            
            state_buffer.append ([ocp_double_pendulum.q1_list[current_state][0], ocp_double_pendulum.q2_list[current_state][0],ocp_double_pendulum.v1_list[current_state][0], ocp_double_pendulum.v2_list[current_state][0]])
            cost_buffer.append(sol.value(ocp_double_pendulum.cost))
        except RuntimeError as e:
            if "Infeasible_Problem_Detected" in str(e):
                print(f"Could not solve for: ")
                print(f"        Initial position (q0): {q0}")
                print(f"        Initial velocity (dq0): {dq0}")
            else:
                print("Runtime error:", e)
        # print("     Starting animation...")
        # animate_double_pendulum(q_trajectory)
        #animate_double_pendulum(q_trajectory)
        # print("     Plot result... ")
        # plot_results(x_sol.T,u_sol.T)
        
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

def animate_double_pendulum(X_opt):
    L1 = config.L1
    L2 = config.L2
    fig, ax = plt.subplots()
    ax.set_xlim(-L1 - L2 - 0.1, L1 + L2 + 0.1)  # Imposta limiti basati sulle lunghezze
    ax.set_ylim(-L1 - L2 - 0.1, L1 + L2 + 0.1)
    line, = ax.plot([], [], 'o-', lw=2)

    def init():
        line.set_data([], [])
        return line,

    def update(frame):
        q1 = X_opt[frame, 0]  # Angolo del primo pendolo
        q2 = X_opt[frame, 1]  # Angolo del secondo pendolo
        x1 = L1 * np.sin(q1)
        y1 = -L1 * np.cos(q1)
        x2 = x1 + L2 * np.sin(q2)
        y2 = y1 - L2 * np.cos(q2)
        line.set_data([0, x1, x2], [0, y1, y2])  # Aggiorna le coordinate della linea
        return line,

    ani = FuncAnimation(fig, update, frames=len(X_opt), init_func=init, blit=True)
    plt.show()
    
if __name__ == "__main__":
    time_start = clock()
    with_N = True
    with_M = False
    
    save_result_bool = True
    train_nn    = True
    
    print("START THE PROGRAM:")
    print(f"Setup choice: N={config.N_step}, M={config.M_step}, tau_min and max={config.TAU_MAX}, max_iter={config.max_iter_opts}")
    print(f"boolean value: with_N={with_N}, with_M={with_M}, save_result={save_result_bool}, train_nn={train_nn}")
    print("press a button to continue")
    input()
    filename = 'dataset/ocp_dataset_DP_train.csv'
    
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
        torch.save( {'model':nn.state_dict()}, "models/model.pt")
    
    print("Total script time:", clock() - time_start)











