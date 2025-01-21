from mpc_double_pendulum import DoublePendulumMPC
from conf_double_pendulum import *
from ocp_double_pendulum import *
'''
Here you can tested all 3 cases showed on Porject A
case are: 
    1. horizon M, NO terminal cost
    2. horizon M + NN as terminal cost
    3. horizon N + M without terminal cost

after is tested the case between
    4. horizon M + NN as terminal cost
    5. horizon M + classic terminal cost
    6. horizon M + hybrid cost
'''


#####################da sistemare1!!!!!!


if __name__ == "__main__":
    
    time_start = clock()
    with_N  = True
    with_M  = True
    mpc_run = True
    term_cost_classic=True
    term_cost_NNet=True
    term_cost_hybrid=True
    
    see_simulation = True
    
    filename_ocp = 'dataset/ocp_dataset_DP_train.csv'
    mpc_double_pendulum = DoublePendulumMPC(filename_ocp)
    nn = NeuralNetwork(filename_ocp,mpc_double_pendulum.nx)
    mpc_double_pendulum.set_terminal_cost(nn)
    
    #STATES_CONSIDER
    config_1 = np.array([-np.pi, -np.pi*2, 0.050, 0.10])
    config_2 = np.array([np.deg2rad(30), np.deg2rad(30), 0.50, 0.10]) 
    config_3 = np.array([np.deg2rad(-30), np.deg2rad(-30), 0.50, 0.10])
    total_config = [config_1, config_2, config_3]
    print("SIMULATION IS READY TO START:")
    print(f"Setup choice: N={config.N_step}, M={config.M_step}, tau_min and max={config.TAU_MAX}, max_iter={config.max_iter_opts}")
    print(f"boolean value: with_N={with_N}, with_M={with_M}, mpc_run={mpc_run}")
    print("PRESS A BUTTON TO CONTINUE")
    input()
    
    for config_init_state in total_config:
        # type of configuration
        counter_config = 1
        #FIRST case --> M without terminal cost
        filename_mpc = f'save_results/config_{counter_config}/config_{counter_config}_results_mpc_M.npz'
        mpc_double_pendulum.simulation(config_initial_state = config_init_state,see_simulation=see_simulation)
        mpc_double_pendulum.save_result_mpc(filename_mpc)
        
        print("finish  M without terminal cost and save result")
        
        #SECOND case --> M + NN as terminal cost
        filename_mpc = f'save_results/config_{counter_config}/config_{counter_config}_results_mpc_M_NN.npz'
        mpc_double_pendulum.simulation(config_initial_state = config_init_state,see_simulation=see_simulation,term_cost_NN_=term_cost_NNet)
        mpc_double_pendulum.save_result_mpc(filename_mpc)
        print("finish  M + NN as terminal cost and save result")
        
        #THIRD case --> N + M without terminal cost
        filename_mpc = f'save_results/config_{counter_config}/config_{counter_config}_results_mpc_M_N.npz'
        mpc_double_pendulum.simulation(with_N,config_initial_state = config_init_state,see_simulation=see_simulation)
        mpc_double_pendulum.save_result_mpc(filename_mpc)
        print("finish  N + M without terminal cost and save result")
        
        
        #--------------------------SPECIAL CASE:-------------------------------
        
        #FOURTH case --> M + NN as terminal cost --see before
        
        #FIVETH case --> M + classic terminal cost
        filename_mpc = f'save_results/config_{counter_config}/config_{counter_config}_results_mpc_M_terminal_cost_standard.npz'
        mpc_double_pendulum.simulation(config_initial_state = config_init_state,see_simulation=see_simulation)
        mpc_double_pendulum.save_result_mpc(filename_mpc)
        print("finish M + classic terminal cost and save result")
        
        #SIXTH case -->  M + hybrid cost        
        filename_mpc = f'save_results/config_{counter_config}/config_{counter_config}_results_mpc_M_hybrid.npz'
        mpc_double_pendulum.simulation(config_initial_state = config_init_state,see_simulation=see_simulation,term_cost_c_=term_cost_classic)
        mpc_double_pendulum.save_result_mpc(filename_mpc)
        print("finish M + hybrid cost and save result")
        
        
    print("Total script time:", clock() - time_start)