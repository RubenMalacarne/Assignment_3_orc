from example_robot_data.robots_loader import load
from adam.casadi.computations import KinDynComputations
import numpy as np
import casadi as cs

from utils.robot_simulator import RobotSimulator
from utils.robot_loaders import loadUR
from utils.robot_wrapper import RobotWrapper
import conf_double_pendulum as config

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
        
        self.N  = config.N
        self.q_des = config.q_des
        self.nx = 2 * self.nq
        self.dt = config.dt
        
        
        self.w_p = config.w_p
        self.w_v = config.w_v
        self.w_a = config.w_a
        self.w_final = config.w_final
        self.number_init_state = number_init_state_
        self.q1_list,self.v1_list,self.q2_list,self.v2_list = self.set_initial_state_list()
        
        self.opti = cs.Opti()
        self.inv_dyn,self.dynamic_f = self.set_dynamics()
        
        print("Initialized DoublePendulumOCP")
    
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
        qs = np.arange(q_min, q_max + q_step, q_step).reshape(n_qs, 1)
        dqs = np.arange(dq_min, dq_max + dq_step, dq_step).reshape(n_dqs, 1)

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

    
    def setup_optimization(self, q0, dq0, q_des):
        
        param_x_init = self.opti.parameter(self.nx)
        param_q_des  = self.opti.parameter(self.nq)
        
        self.opti.set_value(param_x_init, np.concatenate([q0, dq0]))
        self.opti.set_value(param_q_des, q_des)
        
        inv_dyn = self.inv_dyn
        dynamic_f = self.dynamic_f
        
        X, U = [], []
        for i in range(self.N + 1):
            X.append(self.opti.variable(self.nx))
        for i in range(self.N):
            U.append(self.opti.variable(self.nq))
   
        running_cost = 0.0
        for i in range(self.N):
            running_cost += X[i][self.nq:].T @ X[i][self.nq:]  # Velocity cost
            running_cost += X[i][:self.nq].T @ X[i][:self.nq]  # Position cost
            running_cost += U[i].T @ U[i]                      # Acceleration cost

            x_next = X[i] + self.dt * dynamic_f(X[i], U[i])
            self.opti.subject_to(X[i + 1] == x_next)

            tau = inv_dyn(X[i], U[i])
            self.opti.subject_to(self.opti.bounded(config.TAU_MIN, tau, config.TAU_MAX))

        self.opti.subject_to(X[0] == param_x_init)

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
        return sol,optimal_cost
    
    def simulation(self, number_init_state=101):
        state_buffer = []       # Buffer to store initial states
        cost_buffer = []        # Buffer to store optimal costs
        #simulation for each type of the initial state
        for current_state in range(number_init_state):
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
    
    def save_result(self, state_buffer, cost_buffer,booltrain_file = True):
        if (booltrain_file):
            filename = 'ocp_dataset_DP_train.csv'
            positions_q1 = [state[0] for state in state_buffer]
            velocities_v1 = [state[1] for state in state_buffer]
            positions_q2 = [state[2] for state in state_buffer]
            velocities_v2 = [state[3] for state in state_buffer]
            df = pd.DataFrame({'q1': positions_q1, 'v1': velocities_v1, 'q2': positions_q2, 'v2': velocities_v2, 'cost': cost_buffer})
            df.to_csv(filename, index=False)
        else:
            filename = 'ocp_dataset_DP_eval.csv'
            positions_q1 = [state[0] for state in state_buffer]
            velocities_v1 = [state[1] for state in state_buffer]
            positions_q2 = [state[2] for state in state_buffer]
            velocities_v2 = [state[3] for state in state_buffer]
            df = pd.DataFrame({'q1': positions_q1, 'v1': velocities_v1, 'q2': positions_q2, 'v2': velocities_v2, 'cost': cost_buffer})
            df.to_csv(filename, index=False)
        
if __name__ == "__main__":
    time_start = clock()
    ocp = DoublePendulumOCP()
    state_buffer,cost_buffer = ocp.simulation(config.number_init_state)
    
    print("Total script time:", clock() - time_start)
    ocp.save_result(state_buffer,cost_buffer,config.booltrain_file)












