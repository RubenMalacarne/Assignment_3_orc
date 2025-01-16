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
        self.q1_list,self.v1_list,self.q2_list,self.v2_list = self.set_initial_state_list()
        self.N  = config.N_step
        self.M  = config.M_step
        
        
        self.running_costs = [None] * (self.N)
        
        self.q_des = config.q_des
        self.nx = 2 * self.nq
        
        self.w_p = config.w_p
        self.w_v = config.w_v
        self.w_a = config.w_a
        self.w_final = config.w_final
        
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
        M = self.kinDyn.mass_matrix_fun()(H_b, q)[6:, 6:]       
        h = self.kinDyn.bias_force_fun()(H_b, q, v_b, dq)[6:]
        tau = M @ ddq + h       # out control input tau = M(q)*ddq + h(q,dq)
        
        inv_dyn = cs.Function('inv_dyn', [state, ddq], [tau])

        return inv_dyn,dynamic_f

    def NN_cost_pred(self, init_cond):
        # return the rescaled output of the N.N.
        NN_out = self.neural_network.nn_func(init_cond)
        return (NN_out+1.)/2. * (self.out_max - self.out_min) + self.out_min
    
    def setup_mpc(self,sol,optimal_cost,x,X,U,simu,iteration,param_x_init):
        print("     Start the MPC loop")
        #MPC LOOP TO solve the problem
        for i in range(self.N):
            self.opti.set_value(param_x_init, x)
            if (True):
                for k in range(iteration): #initial stat form x[0] until x[M]
                    self.opti.set_initial( X[k] , sol.value(X[k+1])) 
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
            # print ("MCP loop", i , "Comp. time %.3f s"%(stop_time-start_time_), 
            #         "Track err: %.3f"%np.linalg.norm(x[:self.nq]-self.q_des),
            #         "Iters ", sol.stats()["iter_count"],
            #         "return status", sol.stats()["return_status"],
            #         # "dq %.3f"%np.linalg.nomr(x[nq:])
            #         )
            #         #"  %.3f"%np.linalg) 
            running_cost = sol.value(self.running_costs[i])
            print(f"       -Step {i}: Running cost = {running_cost:.4f}")
            # tau = self.inv_dyn(sol.value(X[0]), sol.value(U[0])).toarray().squeeze()
            # if(config.SIMULATOR=="mujoco"):
            #     # do a proper simulation with Mujoco
            #     simu.step(tau, self.dt)
            #     x = np.concatenate([simu.data.qpos, simu.data.qvel])
            # elif(config.SIMULATOR=="pinocchio"):
            #     # do a proper simulation with Pinocchio
            #     simu.simulate(tau, self.dt, int(self.dt/self.dt_sim))
            #     x = np.concatenate([simu.q, simu.v])
            # elif(config.SIMULATOR=="ideal"):
            #     # use state predicted by the MPC as next state
            #     x = sol.value(X[1]) #sample of the next state
            #     simu.display(x[:self.nq]) 
        return sol,optimal_cost
    
    def setup_ocp(self, q0, dq0, q_des, with_N = True, with_M= False, with_terminal_cost=False, mcp_check=False):
        self.opti = cs.Opti()
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
        
        cost = 0.0
        for i in range(iteration):
            cost += X[i][:self.nq].T @ X[i][:self.nq]  # Position cost
            cost += X[i][self.nq:].T @ X[i][self.nq:]  # Velocity cost
            
            cost += U[i].T @ U[i]                      # Acceleration cost
            
            self.running_costs[i]=cost
            
            x_next = X[i] + config.dt * dynamic_f(X[i], U[i])
            self.opti.subject_to(X[i + 1] == x_next)

            tau = inv_dyn(X[i], U[i])
            self.opti.subject_to(self.opti.bounded(config.TAU_MIN, tau, config.TAU_MAX))
            #not add other inequelity constraint!
        #if true, add also the terminal_cost
        if (with_terminal_cost):
            terminal_cost_expr = self.NN_cost_pred(X[-1][:self.nx])
            cost += self.w_final * terminal_cost_expr.T @ terminal_cost_expr

        self.opti.subject_to(X[0] == param_x_init)

        self.opti.minimize(cost)
    
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
        final_cost  = self.opti.value(cost)
        
        #recover the solution
        x_sol = np.array([sol.value(X[k]) for k in range(iteration+1)]).T
        u_sol = np.array([sol.value(U[k]) for k in range(iteration)]).T
        q_trajectory = np.array([sol.value(X[i][:self.nq]) for i in range(iteration + 1)])
        
        q_sol = x_sol[:self.nq,:]
        dq_sol = x_sol[self.nq:,:]
        
        
        tau = np.zeros((self.nq, iteration))
        for i in range(iteration):
            tau[:,i] = inv_dyn(x_sol[:,i], u_sol[:,i]).toarray().squeeze()
            
        final_q =  q_sol[:,iteration]
        final_dq= dq_sol[:,iteration]

        if (mcp_check): 
            simu = None
            # if(config.SIMULATOR=="mujoco"): #not working so well
            #     from orc.utils.mujoco_simulator import MujocoSimulator
            #     print("Creating simulator...")
            #     simu = MujocoSimulator("double_pendulum", self.dt_sim)
            #     simu.set_state(q0, dq0)
            # else:
            #     r = RobotWrapper(self.robot.model, self.robot.collision_model, self.robot.visual_model)
            #     simu = RobotSimulator(config, r)
            #     simu.init(q0, dq0)
            #     simu.display(q0)
            self.setup_mpc(sol,cost,x,X,U,simu,iteration,param_x_init)
        return sol,final_cost ,x_sol, u_sol,final_q,final_dq,q_trajectory
    
    def simulation(self,with_N = True, with_M= False,with_terminal_cost_=False,mcp_check_ = False):
        number_init_state=config.n_init_state_ocp
        state_buffer = []       # Buffer to store initial states
        cost_buffer = []        # Buffer to store optimal costs
        #simulation for each type of the initial state
        for current_state in range(number_init_state):
            q0 = np.array([self.q1_list[current_state][0], self.q2_list[current_state][0]])
            dq0 = np.array([self.v1_list[current_state][0], self.v2_list[current_state][0]])
            print("____________________________________________________________")
            print(f"Start computation OCP... Configuration {current_state+1}:")
            sol,final_cost,x_sol, u_sol,final_q,final_dq,q_trajectory = self.setup_ocp(q0, dq0, self.q_des,with_N,with_M,with_terminal_cost = with_terminal_cost_,mcp_check = mcp_check_)
            print(f"        Initial position (q0): {q0}")
            print(f"        Initial velocity (dq0): {dq0}")
            print(f"        Desired postiion (q):  ", self.q_des)
            print(f"        Final position (q): {final_q}")
            print(f"        Final velocity (dq): {final_dq}")
            print(f"        Final_cost {current_state+1}: ",final_cost)
            
            state_buffer.append ([self.q1_list[current_state][0], self.q2_list[current_state][0],self.v1_list[current_state][0], self.v2_list[current_state][0]])
            cost_buffer.append(final_cost)

            # print("     Starting animation...")
            # animate_double_pendulum(q_trajectory)
            
            # print("     Plot result... ")
            # plot_results(x_sol.T,u_sol.T)
            print ("____________________________________________________________")
            
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

def animate_double_pendulum(X_opt):
    fig, ax = plt.subplots()
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    line, = ax.plot([], [], 'o-', lw=2)

    def init():
        line.set_data([], [])
        return line,

    def update(frame):
        q1 = X_opt[frame, 0]
        q2 = X_opt[frame, 1]
        x1 = np.sin(q1)
        y1 = -np.cos(q1)
        x2 = x1 + np.sin(q2)
        y2 = y1 - np.cos(q2)
        line.set_data([0, x1, x2], [0, y1, y2])
        return line,

    ani = FuncAnimation(fig, update, frames=len(X_opt), init_func=init, blit=True)
    plt.show()

def plot_results(X_opt, U_opt):
    t = np.linspace(0, 10, len(U_opt) + 1)  # Adjusted to match the length of X_opt
    q1 = X_opt[:, 0]
    q2 = X_opt[:, 1]
    dq1 = X_opt[:, 2]
    dq2 = X_opt[:, 3]

    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(t, q1, label='q1')
    plt.plot(t, q2, label='q2')
    plt.ylabel('Position')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(t, dq1, label='dq1')
    plt.plot(t, dq2, label='dq2')
    plt.ylabel('Velocity')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(t[:-1], U_opt, label='u')  # Adjusted to match the length of U_opt
    plt.ylabel('Control')
    plt.xlabel('Time')
    plt.legend()

    plt.show()

if __name__ == "__main__":
    time_start = clock()
    
    
    with_N = True
    with_M = True
    save_result = True
    mpc_run = True
    do_train_nn = True
    with_terminal_cost_ = True
    
    print("START THE PROGRAM:")
    print(f"Setup choice: N={config.N_step}, M={config.M_step}, tau_min and max={config.TAU_MAX}, max_iter={config.max_iter_opts}")
    print(f"boolean value: with_N={with_N}, with_M={with_M}, save_result={save_result}, mpc_run={mpc_run}, do_train_nn={do_train_nn}, with_terminal_cost_={with_terminal_cost_}")
    print("press a button to continue")
    input()
    filename = 'dataset/ocp_dataset_DP_train.csv'
    
    ocp_double_pendulum = DoublePendulumOCP(filename)
    state_buffer,cost_buffer = ocp_double_pendulum.simulation()
    
    
    if (save_result):
        ocp_double_pendulum.save_result(state_buffer,cost_buffer)
    
    nn = NeuralNetwork(filename,ocp_double_pendulum.nx)
    
    if with_terminal_cost_ and mpc_run:
        ocp_double_pendulum.set_terminal_cost(nn)
        state_buffer, cost_buffer = ocp_double_pendulum.simulation(
            with_terminal_cost_=with_terminal_cost_, mcp_check_=mpc_run
        )
        print("finish mpc with terminal cost and also the mpc")
        
    elif with_terminal_cost_:
        ocp_double_pendulum.set_terminal_cost(nn)
        state_buffer, cost_buffer = ocp_double_pendulum.simulation(with_terminal_cost_=with_terminal_cost_)
        print("finish mpc with terminal cost")
        
    elif mpc_run:
        state_buffer, cost_buffer = ocp_double_pendulum.simulation(
            with_terminal_cost_=with_terminal_cost_, mcp_check_=mpc_run
        )
        print("finish also the mpc")
        
    else:
        print("No action taken")
    
    
    
    print("Total script time:", clock() - time_start)











