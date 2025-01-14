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
    
    def __init__(self, robot_model="double_pendulum",number_init_state_ = config.n_init_state_ocp):

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
        

        
        self.opti = cs.Opti()
        self.inv_dyn,self.dynamic_f = self.set_dynamics()
        
        self.param_x_init = self.opti.parameter(self.nx)
        self.param_q_des  = self.opti.parameter(self.nq)

    
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
            
        start_time = clock()
        x = np.concatenate([q0, dq0])
        
            
        print("     Start the MPC loop")
        #Mcp LOOP TO solve the problem
        for i in range(100):
           
        
            if (True):
                for k in range(self.M_step): #initial stat form x[0] until x[M-1]
                    self.opti.set_initial( X[k] , sol.value(X[k+1])
                                    )  #specify the variable and the value, initialize 
                                                        #the initial point and the next iteration of the newton step
                
                for k in range(self.M_step-1): #initial cotroll inputs from U[0] until U[M-2]
                    self.opti.set_initial (U[k], sol.value(U[k+1]))
                    
                #initialize the last state X[M] and the last control U[M-1]
                self.opti.set_initial (X[self.M_step], sol.value(X[self.M_step]))
                self.opti.set_initial (U[self.M_step-1], sol.value(U[self.M_step-1])) 
                
                #initiliaze dual variables:
                lam_g0 = sol.value(self.opti.lam_g)
                self.opti.set_initial(self.opti.lam_g, lam_g0)
            
            self.opti.set_value(self.param_x_init, x)
            try: 
                sol = self.opti.solve()
            except: 
                #if an exception is thrown (e.g. max number of iteration reached)
                sol = self.opti.debug #recover the last value of the solution /another way is disable the SOLVER_MAX_ITER
            stop_time = clock()
            print ("MCP loop", i , "Comp. time %.3f s"%(stop_time-start_time), 
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
    
    
    # def save_result(self, state_buffer, cost_buffer):
    #     filename = 'ocp_dataset_DP_test.csv'
    #     positions_q1 = [state[0] for state in state_buffer]
    #     velocities_v1 = [state[1] for state in state_buffer]
    #     positions_q2 = [state[2] for state in state_buffer]
    #     velocities_v2 = [state[3] for state in state_buffer]
    #     df = pd.DataFrame({'q1': positions_q1, 'v1': velocities_v1, 'q2': positions_q2, 'v2': velocities_v2, 'cost': cost_buffer})
    #     df.to_csv(filename, index=False)
        
if __name__ == "__main__":
    time_start = clock()
    ocp = DoublePendulumOCP()
    state_buffer,cost_buffer = ocp.simulation()
    
    print("Total script time:", clock() - time_start)
    












