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

import conf_single_pendulum as config
from neural_network_singlependulum import NeuralNetwork
from matplotlib.animation import FuncAnimation

class SinglePendulumMPC:
    def __init__(self, filename):
        
        base_path = os.path.dirname(os.path.abspath(__file__))
        urdf_path = os.path.join(base_path, "Single_pendulum_description", "urdf", "single_pendulum.urdf")
        mesh_dir  = os.path.join(base_path, "Single_pendulum_description", "meshes")

        self.robot = RobotWrapper.BuildFromURDF(urdf_path, mesh_dir)        
        self.kinDyn = KinDynComputations(urdf_path, [s for s in self.robot.model.names[1:]])
        
        self.nq     = len(self.robot.model.names[1:])
        
        self.number_init_state = config.n_init_state_ocp
        
        self.q_list, self.v_list = self.set_initial_state_list()
        
        self.N  = config.N_step
        self.M  = config.M_step
        self.N_sim = config.N_sim
        
        
        self.q_des  = config.q_des
        self.nx     = 2 * self.nq   #2x1 for single pendulum

        self.w_p     = config.w_p
        self.w_v     = config.w_v
        self.w_a     = config.w_a
        self.w_final = config.w_final
        self.w_value_nn = config.w_value_nn
        
        self.inv_dyn, self.dynamic_f = self.set_dynamics()
        
        self.filename = filename
        
        print("Initialization SinglePendulumMPC complete!")

    def set_initial_state_list(self):
        n_qs = self.number_init_state
        n_dqs = self.number_init_state
        
        q_min = 0.0
        q_max = np.pi
        
        dq_min = 0.0
        dq_max = 8.0
        
        q1_list = np.zeros((self.number_init_state, 1))
        v1_list = np.zeros((self.number_init_state, 1))
        
        if config.random_initial_set:
            # random points
            q1_list = np.random.uniform(q_min,  q_max,  (self.number_init_state, 1))
            v1_list = np.random.uniform(dq_min, dq_max, (self.number_init_state, 1))
        else:
            # simple way
            q_lin  = np.linspace(q_min, q_max, n_qs).reshape(n_qs, 1)
            dq_lin = np.linspace(dq_min, dq_max, n_dqs).reshape(n_dqs, 1)
            for i in range(self.number_init_state):
                
                q1_list[i, 0] = q_lin[i]
                v1_list[i, 0] = dq_lin[i]
        return q1_list, v1_list


    def set_dynamics(self):
        #use the alternative multi-body dynamics modeling 
        q   = cs.SX.sym("q",  self.nq) #position
        dq  = cs.SX.sym("dq", self.nq) #velocity
        ddq = cs.SX.sym("ddq",self.nq) #acceleration is our control input --> because we use the inverse dynamic problem
        
        state = cs.vertcat(q, dq)       
        rhs   = cs.vertcat(dq, ddq)    
        
        dynamic_f = cs.Function('f', [state, ddq], [rhs])#dynamic function, take in input the state and ddq"=u" (x, u = ddq) and compute dx =(dq,ddq)
        #inverse dynamic function with casadi
        H_b = cs.SX.eye(4)
        v_b = cs.SX.zeros(6)
        
        M = self.kinDyn.mass_matrix_fun()(H_b, q)[6:, 6:] # excluding the first 6 elements 
        #(which usually represent the degrees of freedom of the basic rigid body, such as translations and rotations if the robot is floating).
        h = self.kinDyn.bias_force_fun()(H_b, q, v_b, dq)[6:] # excluding the base DoF
        
        tau = M @ ddq + h    # out control input tau = M(q)*ddq + h(q,dq)
        inv_dyn = cs.Function('inv_dyn', [state, ddq], [tau])
        
        return inv_dyn, dynamic_f

    def set_terminal_cost(self, nn):
        self.neural_network = nn
        self.neural_network.create_casadi_function(
            "single_pendulum",  
            "models/",          
            self.nx,           
            load_weights=True
        )
        print("[MPC] Terminal cost from NN: set up done.")
    
    def cost_from_NN(self, x_state):
        cost_pred = self.neural_network.nn_func(x_state)
        return cost_pred

    def setup_mpc(self, q_des, see_simulation=False,
                  with_N=False, with_M=True,
                  term_cost_c=False,
                  term_cost_NN=False,
                  term_cost_hy=False):
        q_total_trajectory=[]
        dq_total=[]
        ddq_total = []
        All_traj_predicted=[]
        #set time horizon
        if with_N and with_M:
            iteration = self.N + self.M
        elif with_N:
            iteration = self.N
        elif with_M:
            iteration = self.M
        else:
            iteration = self.N  

        self.opti = cs.Opti()
        
        param_x_init = self.opti.parameter(self.nx)
        param_q_des  = self.opti.parameter(self.nq)
        
        inv_dyn = self.inv_dyn
        dynamic_f = self.dynamic_f
        
        X, U = [], []
        for i in range(iteration + 1):
            X.append(self.opti.variable(self.nx))
        for i in range(iteration):
            U.append(self.opti.variable(self.nq))
        
        x_init = np.concatenate([self.q0, self.dq0])
        self.opti.set_value(param_x_init, x_init)
        self.opti.set_value(param_q_des,  q_des)
        
        self.opti.subject_to(X[0] == param_x_init)
        
        self.running_costs = [None] * (iteration)
        cost_expr = 0.0
        for i in range(iteration):
            q_error = X[i][:self.nq] - param_q_des
            dq_error = X[i][self.nq:]
            
            cost_expr += self.w_p * (q_error.T @ q_error)
            cost_expr += self.w_v * (dq_error.T @ dq_error)
            cost_expr += self.w_a * (U[i].T @ U[i])                                  
            self.running_costs[i]=cost_expr
            
            x_next = X[i] + config.dt * dynamic_f(X[i], U[i])
            self.opti.subject_to(X[i + 1] == x_next)

            tau = inv_dyn(X[i], U[i])
            self.opti.subject_to(self.opti.bounded(config.TAU_MIN, tau, config.TAU_MAX))
            #not add other inequelity constraint!
        
        # Terminal cost classic 
        if term_cost_c:
            q_error_final = X[-1][:self.nq] - param_q_des
            dq_final      = X[-1][self.nq:]
            cost_expr    += self.w_final * q_error_final.T @ q_error_final
            cost_expr    += self.w_final * dq_final.T @ dq_final
            dq_final = X[iteration][self.nq:]
            self.opti.subject_to(dq_final == 0.0)
        
        # Terminal cost with NN
        if term_cost_NN:
            cost_pred_nn = self.cost_from_NN(X[-1])  # x[-1] in CasADi
            dq_final      = X[-1][self.nq:]
            cost_expr   += self.w_value_nn * cost_pred_nn
            self.opti.subject_to(dq_final == 0.0)
        
        # Terminal cost "Hybrid"
        if term_cost_hy:
            q_error_final = X[-1][:self.nq] - param_q_des
            dq_final      = X[-1][self.nq:]
            cost_expr    += self.w_final * q_error_final.T @ q_error_final
            cost_expr    += self.w_final * dq_final.T @  dq_final
            cost_pred_nn  = self.cost_from_NN(X[-1])
            cost_expr    += self.w_value_nn * cost_pred_nn
            self.opti.subject_to(dq_final == 0.0)

        self.opti.minimize(cost_expr)
        if (term_cost_NN):
            #se usi la NN
            opts = {
                "error_on_fail": False,
                "ipopt.print_level": 0,
                "ipopt.tol": 1e-1,
                "ipopt.constr_viol_tol": 1e-2,
                "ipopt.compl_inf_tol": 1e-2,
                "print_time": 0,
                "detect_simple_bounds": True,
                "ipopt.max_iter": config.SOLVER_MAX_ITER,
                "ipopt.hessian_approximation": "limited-memory",
                # "ipopt.mu_strategy" : "adaptive"        #use to helt the convergences in NN
            }
        else:
        #per il resto usare questa
        
            opts = {
                "error_on_fail": False,
                "ipopt.print_level": 0,
                "ipopt.tol":  1e-4,
                "ipopt.constr_viol_tol":  1e-4,
                "ipopt.compl_inf_tol":  1e-4,
                "print_time": 0,                # print information about execution time
                "detect_simple_bounds": True,
                "ipopt.max_iter": config.SOLVER_MAX_ITER,   #1000 funziona sicuro       # max number of iteration
                "ipopt.hessian_approximation": "limited-memory"
            }
        if(iteration>30):
            #caso con M e N ho degli out of range quindi usato la stessa che si è usata in OCP
 
            opts = {
                "ipopt.print_level":    0,
                "ipopt.max_iter":       config.max_iter_opts,
                "print_time":           0,
                "ipopt.tol":            1e-3,
                "ipopt.constr_viol_tol":1e-6
            }
            
        self.opti.solver("ipopt", opts)
        
        sol = self.opti.solve()
        final_cost_ocp = self.opti.value(cost_expr)
        
        x_sol = np.array([sol.value(X[k]) for k in range(iteration+1)]).T
        u_sol = np.array([sol.value(U[k]) for k in range(iteration)]).T
        # q_trajectory = np.array([sol.value(X[i][:self.nq]) for i in range(iteration + 1)])
        simu=None
        r = RobotWrapper(self.robot.model, self.robot.collision_model, self.robot.visual_model)
        config.use_viewer = see_simulation
        simu = RobotSimulator(config, r)
        simu.init(self.q0, self.dq0)
        simu.display(self.q0)
        
        
        cost_zero_counter = 0
        t_start_mpc = clock()
        print("Start the MPC loop ...")
        
        previous_cost = None
        cost_zero_counter = 0 
        x = x_init.copy()
        for i_sim in range(self.N_sim):
            
            self.opti.set_value(param_x_init, x)
            
            # Warmstart
            for k in range(iteration):
                self.opti.set_initial(X[k], sol.value(X[k+1]))
            for k in range(iteration-1):
                self.opti.set_initial(U[k], sol.value(U[k+1]))
            self.opti.set_initial(X[iteration], sol.value(X[iteration]))
            self.opti.set_initial(U[iteration-1], sol.value(U[iteration-1]))
            
            lam_g0 = sol.value(self.opti.lam_g)
            self.opti.set_initial(self.opti.lam_g, lam_g0)
            
            try:
                sol = self.opti.solve()
            except:
                sol = self.opti.debug  # fallback
                
            running_cost = self.opti.value(cost_expr)
            

            if (abs(running_cost) < 1e-4) or (previous_cost is not None and abs(running_cost - previous_cost) < 1e-4):
                cost_zero_counter += 1
            else:
                cost_zero_counter = 0 


            if cost_zero_counter >= 3:
                print(f"STOP_CONDITION: Cost below 1e-4 for 5 consecutive iterations")
                break
            previous_cost = running_cost
            
            
            tau = self.inv_dyn(sol.value(X[0]), sol.value(U[0])).toarray().squeeze()
            #to run the visual simulation
            
            if(config.SIMULATOR=="pinocchio"):
                # do a proper simulation with Pinocchio
                simu.simulate(tau, self.dt, int(self.dt/self.dt_sim))
                x = np.concatenate([simu.q, simu.v])
            elif(config.SIMULATOR=="ideal"):
                # use state predicted by the MPC as next state
                x = sol.value(X[1]) #sample of the next state
                simu.display(x[:self.nq]) 
            
            #now recover the next trajectory predicted
            All_traj_predicted.append(np.array([sol.value(X[i][:self.nq]) for i in range(iteration + 1)]))
            actual_q = np.array([sol.value(X[0][:self.nq])])  
            q_total_trajectory.append(actual_q)
            actual_dq = np.array([sol.value(X[0][:self.nq:])])  
            dq_total.append(actual_dq)
            actual_ddq = sol.value(U[0])
            ddq_total.append(actual_ddq)
            store_iteration = i_sim
            print(f"   Step {i_sim} => Running cost = {running_cost:.4f} and actual_q = {actual_q}")
            
        t_mpc = clock() - t_start_mpc
        
        q_sol = x_sol[:self.nq,:]
        dq_sol= x_sol[self.nq:,:]
        q_final   = q_sol[:, -1]
        dq_final  = dq_sol[:, -1]
        All_traj_predicted = np.array(All_traj_predicted)
        tot_iteration = store_iteration
        return (sol, running_cost, q_final, dq_final, x_sol, u_sol, q_total_trajectory, All_traj_predicted,dq_total,ddq_total,t_mpc,tot_iteration)

    def simulation(self, with_N_=False, with_M_=True,
                   config_initial_state=None,
                   see_simulation=False,
                   term_cost_c_=False,
                   term_cost_NN_=False,
                   term_cost_hy_=False):
        print("=== Simulation parameters ===")
        print(f"  with_N_            = {with_N_}")
        print(f"  with_M_            = {with_M_}")
        print(f"  config_initial_state = {config_initial_state}")
        print(f"  see_simulation     = {see_simulation}")
        print(f"  term_cost_c_       = {term_cost_c_}")
        print(f"  term_cost_NN_      = {term_cost_NN_}")
        print(f"  term_cost_hy_      = {term_cost_hy_}")
        print("============================")
        # breakpoint()
        number_init_state = config.n_init_state_ocp
        if (config_initial_state is not None and config_initial_state.size > 0):
            number_init_state = 1
        
        self.state_buffer = []
        self.cost_buffer  = []
        
        for current_state in range(number_init_state):
            self.q0 = np.array([self.q_list[current_state][0]], dtype=np.float64) 
            self.dq0= np.array([self.v_list[current_state][0]], dtype=np.float64)
            if (config_initial_state is not None and config_initial_state.size > 0):
                
                self.q0 = np.array([config_initial_state[0]], dtype=np.float64) 
                self.dq0= np.array([config_initial_state[1]], dtype=np.float64)
            
            print("____________________________________________________________")
            print(f"Start computation MPC... Configuration {current_state+1}:")
            self.sol,self.final_cost,self.final_q,self.final_dq ,self.x_sol, self.u_sol,self.q_total_trajectory,self.All_traj_predicted,self.dq_total,self.ddq_total,self.t_mpc,self.tot_iteration = self.setup_mpc(self.q_des,see_simulation,with_N=with_N_,with_M=with_M_,term_cost_c = term_cost_c_,term_cost_NN=term_cost_NN_,term_cost_hy=term_cost_hy_)
            print(f"        Initial position (q0): {self.q0}")
            print(f"        Initial velocity (dq0): {self.dq0}")
            print(f"        Desired postiion (q):  ", self.q_des)
            print(f"        Final position (q): {self.final_q}")
            print(f"        Final velocity (dq): {self.final_dq}")
            print(f"        Final_cost {current_state+1}: ",self.final_cost)
            print(f"        time {current_state+1}: ",self.t_mpc)
            
            self.state_buffer.append ([self.q_list[current_state][0],self.v_list[current_state][0],])
            self.cost_buffer.append(self.final_cost)
            print ("____________________________________________________________")
    
    def save_result_mpc(self, save_filename="results_mpc/results_mpc_test.npz"):
        os.makedirs(os.path.dirname(save_filename), exist_ok=True)
        data_to_save = {
            "sol": self.sol.stats()["return_status"],
            "final_cost": self.final_cost,
            "final_q": self.final_q,
            "final_dq": self.final_dq,
            "x_sol": self.x_sol,
            "u_sol": self.u_sol,
            "q_trajectory": self.q_total_trajectory,
            "All_traj_predicted":self.All_traj_predicted,
            "t_mpc": self.t_mpc,
            "q0": self.q0,
            "dq0": self.dq0,
            "dq_total":self.dq_total,
            "ddq_total":self.ddq_total,
            "tot_iteration" : self.tot_iteration
        }
        np.savez_compressed(save_filename, **data_to_save)
        print(f"Data saved in: {save_filename}")

# ---------------------------------------------------------------------
#          MAIN(example)
# ---------------------------------------------------------------------
    
if __name__ == "__main__":
    time_start = clock()
    with_N  = True
    with_M  = False
    mpc_run = True
    with_terminal_cost_NN = False
    see_simulation_ = True
    print("START THE PROGRAM:")
    print(f"Setup choice: N={config.N_step}, M={config.M_step}, tau_min and max={config.TAU_MAX}, max_iter={config.max_iter_opts}")
    print(f"boolean value: with_N={with_N}, with_M={with_M}, mpc_run={mpc_run}, with_terminal_cost_={with_terminal_cost_NN}, simulation display{see_simulation_}")
    print("press a button to continue")
    input()
    
    filename =  config.csv_train
    
    nn = NeuralNetwork(
        file_name   = filename,
        input_size  = 2,
        hidden_size = 12,
        output_size = 1,
        n_epochs    = 100,
        batch_size  = 10,
        lr          = 0.0001,
    )

    checkpoint = torch.load("models/model_singlependulum.pt", map_location='cpu')
    nn.load_state_dict(checkpoint['model'])

    mpc_double_pendulum = SinglePendulumMPC(filename)

    mpc_double_pendulum.set_terminal_cost(nn)

    mpc_double_pendulum.simulation(
        term_cost_NN_=with_terminal_cost_NN,
        with_N_=with_N,
        with_M_=with_M,
        see_simulation=see_simulation_
    )
