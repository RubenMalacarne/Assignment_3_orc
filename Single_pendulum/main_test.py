from mpc_single_pendulum import SinglePendulumMPC
from ocp_single_pendulum import SinglePendulumOCP,simulation,save_result
from neural_network_singlependulum import NeuralNetwork
import conf_single_pendulum as config
from utility_func import  plot_all_trajectories, animate_all_simulations_together
from utility_func import  animate_plots_together, all_mpc_time, plot_joint_dynamics
import torch
from time import time as clock
import numpy as np
'''
Here you can tested all 3 cases showed on Porject A
case are: 
    1. horizon M, NO terminal cost
    2. horizon M + classic terminal cost
    3. horizon N + M without terminal cost
    4. horizon M + NN as terminal cost
    
''' 
  
if __name__ == "__main__":
    
    time_start = clock()
    OCP_step = True
    NN_step = True
    MPC_step = True
    RESULT_step = True
    
    #STATES_CONSIDER FOR MPC
    config_1 = np.array([-np.pi, 0.050])
    # config_2 = np.array([np.deg2rad(30),0.50]) 
    # config_3 = np.array([np.deg2rad(-30), 0.50])
    total_config = [config_1]#, config_2, config_3]
    
    if (OCP_step):
        #--------------------------------------------
        #                   OCP PART
        #--------------------------------------------
        time_ocp = clock()
        with_N = True
        with_M = False
        save_result_bool = True
        train_nn    = True
        
        filename = config.csv_train
        print("START OCP:")
        print(f"Setup choice:number initial states: {config.n_init_state_ocp}, N={config.N_step}, M={config.M_step}, tau_min and max={config.TAU_MAX}, max_iter={config.max_iter_opts}")
        print(f"boolean value: with_N={with_N}, with_M={with_M}, save_result={save_result_bool}, train_nn={train_nn}, Random distribution {config.random_initial_set}")
        # print("press a button to continue")
        # input()
        ocp_double_pendulum = SinglePendulumOCP(filename)
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

    if (NN_step):
        csv_train = config.csv_train 
        csv_eval = config.csv_eval 
        net = NeuralNetwork(csv_train)
        net.trainig_part()
        net.plot_training_history()

        torch.save({'model': net.state_dict()}, "models/model.pt")
        print("Model saved.")
        net.evaluaunation(csv_eval)
        
    if (MPC_step):
        #--------------------------------------------
        #                   MPC PART
        #--------------------------------------------
        with_N  = False
        with_M  = True
        term_cost_classic=True
        term_cost_NNet=True
        term_cost_hybrid=True
        
        see_simulation = False
        
        filename_ocp = config.csv_train
        mpc_single_pendulum = SinglePendulumMPC(filename_ocp)
        nn = NeuralNetwork(filename_ocp,mpc_single_pendulum.nx)
        mpc_single_pendulum.set_terminal_cost(nn)
        
        print("SIMULATION IS READY TO START:")
        print(f"Setup choice: N={config.N_step}, M={config.M_step}, tau_min and max={config.TAU_MAX}, max_iter={config.max_iter_opts}")
        print(f"boolean value: with_N={with_N}, with_M={with_M}")
        # print("PRESS A BUTTON TO CONTINUE")
        # input()
        counter_config = 0
        
        for config_init_state in total_config:
            # type of configuration
            counter_config += 1
            
            #FIRST case --> M without terminal cost
            filename_mpc = f'save_results/config_{counter_config}/config_{counter_config}_results_mpc_M.npz'
            mpc_single_pendulum.simulation(config_initial_state = config_init_state,see_simulation=see_simulation)
            mpc_single_pendulum.save_result_mpc(filename_mpc)
            print("finish  M without terminal cost and save result")
            filenpz = "save_results/config_1/config_1_results_mpc_M.npz"
            #SECOND case --> N + M without terminal cost
            filename_mpc = f'save_results/config_{counter_config}/config_{counter_config}_results_mpc_M_N.npz'
            mpc_single_pendulum.simulation(with_N_=with_N,with_M_=with_M,config_initial_state = config_init_state,see_simulation=see_simulation)
            mpc_single_pendulum.save_result_mpc(filename_mpc)
            print("finish  N + M without terminal cost and save result")
            # #THIRD case --> M + classic terminal cost
            filename_mpc = f'save_results/config_{counter_config}/config_{counter_config}_results_mpc_M_terminal_cost_standard.npz'
            mpc_single_pendulum.simulation(with_M_=with_M,config_initial_state = config_init_state,see_simulation=see_simulation,term_cost_c_=term_cost_classic)
            mpc_single_pendulum.save_result_mpc(filename_mpc)
            print("finish M + classic terminal cost and save result")
            # FOURTH case --> M + NN as terminal cost
            filename_mpc = f'save_results/config_{counter_config}/config_{counter_config}_results_mpc_M_NN.npz'
            mpc_single_pendulum.simulation(config_initial_state = config_init_state,see_simulation=see_simulation,term_cost_NN_=term_cost_NNet)
            mpc_single_pendulum.save_result_mpc(filename_mpc)
            print("finish  M + NN as terminal cost and save result")
            # #FIVETH case --> M + Hybrid as terminal cost
            # filename_mpc = f'save_results/config_{counter_config}/config_{counter_config}_results_mpc_M_HY.npz'
            # mpc_single_pendulum.simulation(config_initial_state = config_init_state,see_simulation=see_simulation,term_cost_hy_=term_cost_hybrid)
            # mpc_single_pendulum.save_result_mpc(filename_mpc)
            # print("finish  M + Hybrid as terminal cost and save result")
            
    if (RESULT_step):
        #--------------------------------------------
        #                   RESULT PART
        #--------------------------------------------
        counter_config = 0
        for config_init_state in total_config:
            counter_config += 1
            file_paths = [
            f"save_results/config_{counter_config}/config_{counter_config}_results_mpc_M.npz",
            f"save_results/config_{counter_config}/config_{counter_config}_results_mpc_M_NN.npz",
            f"save_results/config_{counter_config}/config_{counter_config}_results_mpc_M_N.npz",
            f"save_results/config_{counter_config}/config_{counter_config}_results_mpc_M_terminal_cost_standard.npz",
            ]
            plot_all_trajectories(file_paths)
            animate_all_simulations_together(file_paths)
            animate_plots_together(file_paths)
            all_mpc_time(file_paths)
            plot_joint_dynamics(file_paths)

    print("Total script time:", clock() - time_start)
  