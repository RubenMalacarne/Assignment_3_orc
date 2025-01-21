from example_robot_data.robots_loader import load
from adam.casadi.computations import KinDynComputations
import numpy as np
import casadi as cs
import os
from utils.robot_simulator import RobotSimulator
from utils.robot_loaders import loadUR
from utils.robot_wrapper import RobotWrapper
import torch
import pinocchio as pin

import pandas as pd

from time import time as clock
from time import sleep

import matplotlib.pyplot as plt
from termcolor import colored

# IMPORTA IL TUO FILE DI CONFIGURAZIONE DEL SINGOLO PENDOLO
import conf_single_pendulum as config  
from neural_network_singlependulum import NeuralNetwork

from matplotlib.animation import FuncAnimation

class SinglePendulumOCP:
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
        
        self.q_des  = config.q_des
        self.nx     = 2 * self.nq   #2x1 for single pendulum

        self.w_p     = config.w_p
        self.w_v     = config.w_v
        self.w_a     = config.w_a
        self.w_final = config.w_final
        
        self.inv_dyn, self.dynamic_f = self.set_dynamics()
        
        self.filename = filename
        
        print("Inizializzazione SinglePendulumOCP completata!")

    def set_initial_state_list(self):
        n_qs = self.number_init_state
        n_dqs = self.number_init_state
        
        q_min = 0.0
        q_max = np.pi
        
        dq_min = 0.0
        dq_max = 8.0
        
        if config.random_initial_set:
            # random points
            qs  = np.random.uniform(q_min,  q_max,  (n_qs, 1))
            dqs = np.random.uniform(dq_min, dq_max, (n_dqs, 1))
        else:
            # simple way
            qs  = np.linspace(q_min, q_max, n_qs).reshape(n_qs, 1)
            dqs = np.linspace(dq_min, dq_max, n_dqs).reshape(n_dqs, 1)

        return qs, dqs

    def set_dynamics(self):
        #use the alternative multi-body dynamics modeling 
        q   = cs.SX.sym("q",  self.nq) #position
        dq  = cs.SX.sym("dq", self.nq) #velocity
        ddq = cs.SX.sym("ddq",self.nq) #acceleration is our control input --> because we use the inverse dynamic problem
        
        state = cs.vertcat(q, dq)       #vertical concatenation q and dq
        rhs   = cs.vertcat(dq, ddq)    #vertical concatenation dq and ddq
        
        dynamic_f = cs.Function('f', [state, ddq], [rhs])#dynamic function, take in input the state and ddq"=u" (x, u = ddq) and compute dx =(dq,ddq)
        #inverse dynamic function with casadi
        #it's neccessary because in URDF is present a base link
        H_b = cs.SX.eye(4)
        v_b = cs.SX.zeros(6)
        
        M = self.kinDyn.mass_matrix_fun()(H_b, q)[6:, 6:] # excluding the first 6 elements 
        #(which usually represent the degrees of freedom of the basic rigid body, such as translations and rotations if the robot is floating).
        h = self.kinDyn.bias_force_fun()(H_b, q, v_b, dq)[6:] # excluding the base DoF
        
        tau = M @ ddq + h    # out control input tau = M(q)*ddq + h(q,dq)
        inv_dyn = cs.Function('inv_dyn', [state, ddq], [tau])
        
        return inv_dyn, dynamic_f

    def setup_ocp(self, q0, dq0, with_N=True, with_M=False):
        self.opti = cs.Opti()
        
        if with_N and with_M:
            iteration = self.N + self.M
        elif with_N:
            iteration = self.N
        elif with_M:
            iteration = self.M
        else:
            iteration = self.N # default
            
        X, U = [], []
        for i in range(iteration + 1):
            X.append(self.opti.variable(self.nx))
        for i in range(iteration):
            U.append(self.opti.variable(self.nq))
        
        # parameters (not variables!!!): stato iniziale e q_des
        param_x_init = self.opti.parameter(self.nx)
        param_q_des  = self.opti.parameter(self.nq)
        
        # Setting of the value of the parameters
        x_init = np.array([q0, dq0])  
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
            cost_expr += self.w_a * cs.dot(U[i], U[i]) #acceleration lik in input
            
            # dynamic implementation
            x_next = X[i] + config.dt * self.dynamic_f(X[i], U[i])
            self.opti.subject_to(X[i+1] == x_next)
            
            # "tourque" limit
            tau = self.inv_dyn(X[i], U[i])
            self.opti.subject_to(self.opti.bounded(config.TAU_MIN, tau, config.TAU_MAX))
        
        # add terminal cost
        # q_error_final  = X[-1][:self.nq] - param_q_des
        # dq_error_final = X[-1][self.nq:]
        # cost_expr     += self.w_final * cs.dot(q_error_final,  q_error_final)
        # cost_expr     += self.w_final * cs.dot(dq_error_final, dq_error_final)
        
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


def simulation(ocp_single_pendulum, with_N=True, with_M=False):
    """
    Risolve l'OCP per ogni stato iniziale e salva i costi.
    """
    number_init_state = config.n_init_state_ocp
    state_buffer = []      
    cost_buffer = []       
    
    for current_state in range(number_init_state):
        q0  = ocp_single_pendulum.q_list[current_state][0]
        dq0 = ocp_single_pendulum.v_list[current_state][0]
        
        print("____________________________________________________________")
        print(f"Start computation OCP... Configuration {current_state+1}:")
        try:
            sol, final_cost, x_sol, u_sol, q_trajectory, dq_trajectory = \
                ocp_single_pendulum.setup_ocp(q0, dq0, with_N, with_M)
            
            print(f"        Initial position (q0): {q0}")
            print(f"        Initial velocity (dq0): {dq0}")
            print(f"        Desired position (q_des): {ocp_single_pendulum.q_des}")
            print(f"        Final_cost {current_state+1}: {final_cost}")
            
            state_buffer.append([q0, dq0])
            cost_buffer.append(final_cost)
            
            # animate_single_pendulum(q_trajectory) 
            # plot_results(x_sol.T, u_sol.T)
            
        except RuntimeError as e:
            if "Infeasible_Problem_Detected" in str(e):
                print("Problema non risolvibile per:")
                print(f"        q0={q0}, dq0={dq0}")
            else:
                print("Runtime error:", e)
        print ("____________________________________________________________")
    return state_buffer, cost_buffer

def save_result(filename, state_buffer, cost_buffer):
    """
    Salvataggio dei risultati in un CSV, 
    adesso abbiamo solo q e dq invece di q1,q2,v1,v2.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    positions_q  = [s[0] for s in state_buffer]
    velocities_dq = [s[1] for s in state_buffer]
    
    df = pd.DataFrame({
        'q':  positions_q, 
        'dq': velocities_dq, 
        'cost': cost_buffer
    })
    df.to_csv(filename, index=False)
    
    print(f"File salvato: {filename}")

#da sistemare
def animate_single_pendulum(q_trajectory):
    """
    Esempio di animazione per il singolo pendolo,
    assumendo una lunghezza L1.
    """
    L1 = config.L1
    fig, ax = plt.subplots()
    ax.set_xlim(-L1 - 0.1, L1 + 0.1)
    ax.set_ylim(-L1 - 0.1, L1 + 0.1)
    
    line, = ax.plot([], [], 'o-', lw=2)
    
    def init():
        line.set_data([], [])
        return line,

    def update(frame):
        q = -q_trajectory[frame, 0] 
        x1 = L1 * np.sin(q)
        y1 = -L1 * np.cos(q)
        line.set_data([0, x1], [0, y1])
        return line,

    ani = FuncAnimation(fig, update, frames=len(q_trajectory), init_func=init, blit=True)
    plt.show()


# ---------------------------------------------------------------------
#          MAIN(example)
# ---------------------------------------------------------------------
    
if __name__ == "__main__":
    time_start = clock()

    with_N = True
    with_M = False
    save_result_bool = True
    train_nn = True 
    print("START THE PROGRAM:")
    print(f"Setup choice: #init_state={config.n_init_state_ocp}, N={config.N_step}, M={config.M_step}, Tau=[{config.TAU_MIN},{config.TAU_MAX}]")
    print(f"boolean value: with_N={with_N}, with_M={with_M}, save_result={save_result_bool}, train_nn={train_nn}, random_init={config.random_initial_set}")
    print("Premi INVIO per continuare.")
    input()

    filename = 'dataset/ocp_dataset_SP_train.csv'
    
    ocp_single_pendulum = SinglePendulumOCP(filename)
    state_buffer, cost_buffer = simulation(ocp_single_pendulum, with_N, with_M)
    
    if save_result_bool:
        save_result(filename, state_buffer, cost_buffer)
        print("Finito di salvare i risultati.")
        
    if train_nn:
        nn = NeuralNetwork(filename, ocp_single_pendulum.nx)  
        nn.trainig_part()
        nn.plot_training_history()
        
        torch.save({'model': nn.state_dict()}, "models/model_singlependulum.pt")
    
    print("Tempo totale script:", clock() - time_start)
