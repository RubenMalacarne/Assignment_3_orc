from example_robot_data.robots_loader import load
from adam.casadi.computations import KinDynComputations
import numpy as np
import casadi as cs

from utils.robot_simulator import RobotSimulator
from utils.robot_loaders import loadUR
from utils.robot_wrapper import RobotWrapper
import conf_double_pendulum as config
import torch
from neural_network_doublependulum import NeuralNetwork

import pandas as pd

from time import time as clock
from time import sleep

import matplotlib.pyplot as plt
from termcolor import colored


class DoublePendulumOCP:
    
    def __init__(self, robot_model="double_pendulum",number_init_state_ = config.number_init_state):

        self.robot = load(robot_model)
        self.kinDyn = KinDynComputations(self.robot.urdf, [s for s in self.robot.model.names[1:]])
        self.nq = len(self.robot.model.names[1:])
                
        self.M_step      = config.M_step
        self.q_des  = config.q_des
        self.nx     = 2 * self.nq
        self.dt     = config.dt
        self.number_init_state = number_init_state_
        self.w_p    = config.w_p
        self.w_v    = config.w_v
        self.w_a    = config.w_a
        self.w_final= config.w_final
        
        self.q1_list,self.v1_list,self.q2_list,self.v2_list = self.set_initial_state_list()

        # Load the trained neural network
        self.neural_network = NeuralNetwork(file_name="models/ocp_dataset_DP_train.csv",input_size=self.nx)
        
        # self.neural_network.line_stack.load_state_dict(torch.load("models/model.pt"))
        # self.neural_network.line_stack.eval()
        dataset = pd.read_csv('models/ocp_dataset_DP_train.csv')
        self.out_min = min(dataset['cost'])
        self.out_max = max(dataset['cost'])

        f_teminal_cost = self.neural_network.create_casadi_function("double_pendulum",'models/', self.nx, load_weights=True)
    
        self.opti = cs.Opti()
        self.inv_dyn,self.dynamic_f = self.set_dynamics()
        
        self.param_x_init = self.opti.parameter(self.nx)
        self.param_q_des  = self.opti.parameter(self.nq)
    
    def NN_cost_pred(self, init_cond):
        # return the rescaled output of the N.N.
        NN_out = self.neural_network.nn_func(init_cond)
        return (NN_out+1.)/2. * (self.out_max - self.out_min) + self.out_min
    def set_initial_state_list(self):
        # qs1_0,dqs1_0,qs2_0,dqs2_0 =initial_state ()
        n_qs = self.number_init_state
        n_dqs = self.number_init_state
        q_min = 0
        q_max = np.pi
        dq_min = 0.0
        dq_max = 8.0
        #sfasamento
        phi = np.pi / 4  
        dq_phi = 2.0
        
        q_step = (q_max - q_min) / (n_qs - 1)
        dq_step = (dq_max - dq_min) / (n_dqs - 1)
        # qs = np.arange(q_min, q_max + q_step, q_step).reshape(n_qs, 1)
        # dqs = np.arange(dq_min, dq_max + dq_step, dq_step).reshape(n_dqs, 1)
        qs = np.linspace(q_min, q_max, n_qs).reshape(n_qs, 1)
        dqs = np.linspace(dq_min, dq_max, n_dqs).reshape(n_dqs, 1)

        qs2 = qs    #(qs + phi) % (2 * np.pi)
        dqs2 = dqs  #(dqs + dq_phi) % (dq_max - dq_min) + dq_min
        
        return qs,dqs,qs2,dqs2
        
    
    def set_dynamics(self):
        q       = cs.SX.sym("q", self.nq)
        dq      = cs.SX.sym("dq", self.nq)
        ddq     = cs.SX.sym("ddq", self.nq)
        state   = cs.vertcat(q, dq)
        H_b     = cs.SX.eye(4)
        v_b     = cs.SX.zeros(6)
        M = self.kinDyn.mass_matrix_fun()(H_b, q)[6:, 6:]
        h = self.kinDyn.bias_force_fun()(H_b, q, v_b, dq)[6:]
        tau = M @ ddq + h
        
        rhs = cs.vertcat(dq, ddq)
        dynamic_f  = cs.Function('f', [state, ddq], [rhs]) #dynamic function, take in input the state and ddq (x, u = ddq) and compute dx =(dq,ddq)
        inv_dyn = cs.Function('inv_dyn', [state, ddq], [tau])
        return inv_dyn,dynamic_f

    def terminal_cost(self,state):
        return self.neural_network.nn_func(state)
    
    def setup_optimization(self, q0, dq0, q_des):
        
        
        self.opti.set_value(self.param_x_init, np.concatenate([q0, dq0]))
        self.opti.set_value(self.param_q_des, q_des)
        
        inv_dyn = self.inv_dyn
        dynamic_f = self.dynamic_f
        
        X, U = [], []
        for i in range(self.M_step + 1):
            X.append(self.opti.variable(self.nx))
        for i in range(self.M_step):
            U.append(self.opti.variable(self.nq))
   
        running_cost = 0.0
        for i in range(self.M_step):
            running_cost += X[i][self.nq:].T @ X[i][self.nq:]  # Velocity cost
            running_cost += X[i][:self.nq].T @ X[i][:self.nq]  # Position cost
            running_cost += U[i].T @ U[i]                      # Acceleration cost

            x_next = X[i] + self.dt * dynamic_f(X[i], U[i])
            self.opti.subject_to(X[i + 1] == x_next)

            tau = inv_dyn(X[i], U[i])
            self.opti.subject_to(self.opti.bounded(config.TAU_MIN, tau, config.TAU_MAX))
        
        # # Add terminal cost using the neural network
        # breakpoint()
        terminal_cost_expr = self.NN_cost_pred(X[-1][:self.nx])
        running_cost += self.w_final * terminal_cost_expr.T @ terminal_cost_expr
        breakpoint()

        self.opti.subject_to(X[0] == self.param_x_init)
        
        self.opti.minimize(running_cost)
    
        opts = {
            "ipopt.print_level": 0,
            "ipopt.tol": 1e-6,
            "ipopt.constr_viol_tol": 1e-6,
            "ipopt.compl_inf_tol": 1e-6,
            "print_time": 0,
            "detect_simple_bounds": True
        }
        self.opti.solver("ipopt", opts)
        sol = self.opti.solve()
        optimal_cost = self.opti.value(running_cost)
        
        print("Optimization complete. Optimal cost:", optimal_cost)
        
        return sol, optimal_cost
    
            
        
    def simulation(self,BOOL_MCP = True):
        state_buffer = []       # Buffer to store initial states
        cost_buffer = []        # Buffer to store optimal costs
        #simulation for each type of the initial state
        for current_state in range(self.number_init_state-1):
            q0 = np.array([self.q1_list[current_state][0], self.q2_list[current_state][0]])
            dq0 = np.array([self.v1_list[current_state][0], self.v2_list[current_state][0]])

            sol,optimal_cost = self.setup_optimization(q0, dq0, self.q_des)

            print(f"Configuration {current_state+1}:")
            print(f"  Initial position (q0): {q0}")
            print(f"  Initial velocity (dq0): {dq0}")
            print("  Final position (q):", sol.value(self.opti.debug.value(sol.value(self.q_des))))
            print(f"total_cost {current_state+1}: ",optimal_cost)
            
            state_buffer.append ([self.q1_list[current_state][0], self.q2_list[current_state][0],self.v1_list[current_state][0], self.v2_list[current_state][0]])
            cost_buffer.append(optimal_cost)

        return state_buffer,cost_buffer
    
        
if __name__ == "__main__":
    time_start = clock()
    ocp = DoublePendulumOCP()
    state_buffer,cost_buffer = ocp.simulation()
    
    print("Total script time:", clock() - time_start)
    












