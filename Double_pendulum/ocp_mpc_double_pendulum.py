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

class DoublePendulumOCP:
    
    def __init__(self,filename, robot_model="double_pendulum"):
        
        self.robot  = load(robot_model)
        self.kinDyn = KinDynComputations(self.robot.urdf, [s for s in self.robot.model.names[1:]])
        self.nq     = len(self.robot.model.names[1:])
        
        self.number_init_state = config.n_init_state_ocp
        self.q1_list,self.v1_list,self.q2_list,self.v2_list = self.set_initial_state_list()
        self.N  = config.N_step
        self.M  = config.M_step
        
        self.q_des = config.q_des
        self.nx = 2 * self.nq
        self.dt = config.dt
        
        self.w_p = config.w_p
        self.w_v = config.w_v
        self.w_a = config.w_a
        self.w_final = config.w_final
        
        self.opti = cs.Opti()
        self.inv_dyn,self.dynamic_f = self.set_dynamics()
        
        self.filename= filename
        
        print("Initializeation Double_Pendulum OCP complete!")
    
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

    def NN_cost_pred(self, init_cond):
        # return the rescaled output of the N.N.
        NN_out = self.neural_network.nn_func(init_cond)
        return (NN_out+1.)/2. * (self.out_max - self.out_min) + self.out_min
    
    def setup_mpc(self,sol,optimal_cost,x,X,U,simu,iteration,param_x_init):
        start_time_ = clock()
        print("     Start the MPC loop")
        #Mcp LOOP TO solve the problem
        for i in range(100):
            if (True):
                for k in range(iteration): #initial stat form x[0] until x[M-1]
                    self.opti.set_initial( X[k] , sol.value(X[k+1])
                                    )  #specify the variable and the value, initialize 
                                                        #the initial point and the next iteration of the newton step
                
                for k in range(iteration-1): #initial cotroll inputs from U[0] until U[M-2]
                    self.opti.set_initial (U[k], sol.value(U[k+1]))
                    
                #initialize the last state X[M] and the last control U[M-1]
                self.opti.set_initial (X[iteration], sol.value(X[iteration]))
                self.opti.set_initial (U[iteration-1], sol.value(U[iteration-1])) 
                
                #initiliaze dual variables:
                lam_g0 = sol.value(self.opti.lam_g)
                self.opti.set_initial(self.opti.lam_g, lam_g0)
            
            self.opti.set_value(param_x_init, x)
            try: 
                sol = self.opti.solve()
            except: 
                #if an exception is thrown (e.g. max number of iteration reached)
                sol = self.opti.debug #recover the last value of the solution /another way is disable the SOLVER_MAX_ITER
            stop_time = clock()
            print ("MCP loop", i , "Comp. time %.3f s"%(stop_time-start_time_), 
                    "Track err: %.3f"%np.linalg.norm(x[:self.nq]-self.q_des),
                    "Iters ", sol.stats()["iter_count"],
                    "return status", sol.stats()["return_status"],
                    # "dq %.3f"%np.linalg.nomr(x[nq:])
                    )
                    #"  %.3f"%np.linalg)
                    
                #track err indica quanto si discodta dal targhet dato
            tau = self.inv_dyn(sol.value(X[0]), sol.value(U[0])).toarray().squeeze()
            if(config.SIMULATOR=="mujoco"):
                # do a proper simulation with Mujoco
                simu.step(tau, self.dt)
                x = np.concatenate([simu.data.qpos, simu.data.qvel])
            elif(config.SIMULATOR=="pinocchio"):
                # do a proper simulation with Pinocchio
                simu.simulate(tau, self.dt, int(self.dt/self.dt_sim))
                x = np.concatenate([simu.q, simu.v])
            elif(config.SIMULATOR=="ideal"):
                # use state predicted by the MPC as next state
                x = sol.value(X[1]) #sample of the next state
                simu.display(x[:self.nq])
        return sol,optimal_cost
    
    def setup_ocp(self, q0, dq0, q_des, with_N = True, with_M= False, with_terminal_cost=False, mcp_check=True):
        iteration = self.N
        if (with_N) : iteration = self.N
        elif (with_M) : iteration = self.M
        elif (with_N and with_M): iteration = self.N + self.M 
        
        param_x_init = self.opti.parameter(self.nx)
        param_q_des  = self.opti.parameter(self.nq)
        
        x = np.concatenate([q0, dq0])
        self.opti.set_value(param_x_init, x)
        self.opti.set_value(param_q_des, q_des)
        
        inv_dyn = self.inv_dyn
        dynamic_f = self.dynamic_f
        
        X, U = [], []
        for i in range(iteration + 1):
            X.append(self.opti.variable(self.nx))
        for i in range(iteration):
            U.append(self.opti.variable(self.nq))
        
        running_cost = 0.0
        for i in range(iteration):
            running_cost += X[i][:self.nq].T @ X[i][:self.nq]  # Position cost
            running_cost += X[i][self.nq:].T @ X[i][self.nq:]  # Velocity cost
            running_cost += U[i].T @ U[i]                      # Acceleration cost

            x_next = X[i] + self.dt * dynamic_f(X[i], U[i])
            self.opti.subject_to(X[i + 1] == x_next)

            tau = inv_dyn(X[i], U[i])
            self.opti.subject_to(self.opti.bounded(config.TAU_MIN, tau, config.TAU_MAX))
            #not add other inequelity constraint!
        #if true, add also the terminal_cost
        if (with_terminal_cost):
            terminal_cost_expr = self.NN_cost_pred(X[-1][:self.nx])
            running_cost += self.w_final * terminal_cost_expr.T @ terminal_cost_expr

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
        if (mcp_check):
            print("start the visualization of the robot...")
            if(config.SIMULATOR=="mujoco"): #not working so well
                from orc.utils.mujoco_simulator import MujocoSimulator
                print("Creating simulator...")
                simu = MujocoSimulator("double_pendulum", self.dt_sim)
                simu.set_state(q0, dq0)
            else:
                r = RobotWrapper(self.robot.model, self.robot.collision_model, self.robot.visual_model)
                simu = RobotSimulator(config, r)
                simu.init(q0, dq0)
                simu.display(q0)
            
            self.setup_mpc(sol,optimal_cost,x,X,U,simu,iteration,param_x_init)
            
        return sol,optimal_cost
    
   
    def simulation(self, number_init_state=config.n_init_state_ocp,with_terminal_cost_=False,mcp_check_ = False):
        state_buffer = []       # Buffer to store initial states
        cost_buffer = []        # Buffer to store optimal costs
        #simulation for each type of the initial state
        for current_state in range(number_init_state):
            q0 = np.array([self.q1_list[current_state][0], self.q2_list[current_state][0]])
            dq0 = np.array([self.v1_list[current_state][0], self.v2_list[current_state][0]])
            
            sol,optimal_cost = self.setup_ocp(q0, dq0, self.q_des,with_terminal_cost = with_terminal_cost_,mcp_check = mcp_check_)

            print(f"Configuration {current_state+1}:")
            print(f"  Initial position (q0): {q0}")
            print(f"  Initial velocity (dq0): {dq0}")
            print(f"  Final position (q): {sol.value(self.opti.debug.value(sol.value(self.q_des)))}")
            print(f"total_cost {current_state+1}: ",optimal_cost)
            print ("______________________________________________")
            
            state_buffer.append ([self.q1_list[current_state][0], self.q2_list[current_state][0],self.v1_list[current_state][0], self.v2_list[current_state][0]])
            cost_buffer.append(optimal_cost)
        return state_buffer,cost_buffer
     
        
    def save_result(self, state_buffer, cost_buffer):
        
        os.makedirs(os.path.dirname(self.filename), exist_ok=True)

        positions_q1 =  [state[0] for state in state_buffer]
        positions_q2 =  [state[1] for state in state_buffer]
        velocities_v1 = [state[2] for state in state_buffer]
        velocities_v2 = [state[3] for state in state_buffer]
        df = pd.DataFrame({'q1': positions_q1, 'q2': positions_q2, 'v1': velocities_v1, 'v2': velocities_v2, 'cost': cost_buffer})
        df.to_csv(self.filename, index=False)
        
        print(f"File saved: {self.filename}")
    
    def set_terminal_cost(self,nn):
        self.neural_network = nn
        self.neural_network.create_casadi_function("double_pendulum",'models/', self.nx, load_weights=True)
        dataset = pd.read_csv(self.filename)
        self.out_min = min(dataset['cost'])
        self.out_max = max(dataset['cost'])

if __name__ == "__main__":
    time_start = clock()
    do_train_nn = True
    save_result = True
    with_terminal_cost_ = False
    mpc_run = True
    filename = 'dataset/ocp_dataset_DP_train.csv'
    
    ocp_double_pendulum = DoublePendulumOCP(filename)
    state_buffer,cost_buffer = ocp_double_pendulum.simulation()
    
    if (save_result):
        ocp_double_pendulum.save_result(state_buffer,cost_buffer)
    
    nn = NeuralNetwork(filename,ocp_double_pendulum.nx)
    
    if (do_train_nn):
        nn.trainig_part()
        nn.plot_training_history()
        torch.save( {'model':nn.state_dict()}, "models/model.pt")
        print("model_saved!")
    if (with_terminal_cost_):
        ocp_double_pendulum.set_terminal_cost(nn)
        
        state_buffer,cost_buffer = ocp_double_pendulum.simulation(with_terminal_cost_= with_terminal_cost_)
    
    if (mpc_run):
        state_buffer,cost_buffer = ocp_double_pendulum.simulation(with_terminal_cost_= with_terminal_cost_,mcp_check_ =mpc_run )
        print ("finish also the mpc")
    
    
    print("Total script time:", clock() - time_start)











