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
if __name__ == "__main__":
    
    time_start = clock()
    with_N  = False
    with_M  = True
    mpc_run = True
    with_terminal_cost_ = False
    
    see_simulation = False
    
    filename = 'dataset/ocp_dataset_DP_train.csv'
    mpc_double_pendulum = DoublePendulumMPC(filename)
    nn = NeuralNetwork(filename,mpc_double_pendulum.nx)
    #STATES_CONSIDER
    
    config_1 = np.array([-np.pi, -np.pi*2, 0.050, 0.10])
    config_2 = np.array([np.deg2rad(30), np.deg2rad(30), 0.50, 0.10]) 
    config_3 = np.array([np.deg2rad(-30), np.deg2rad(-30), 0.50, 0.10])
    total_config = [config_1, config_2, config_3]
    print("SIMULATION IS READY TO START:")
    print(f"Setup choice: N={config.N_step}, M={config.M_step}, tau_min and max={config.TAU_MAX}, max_iter={config.max_iter_opts}")
    print(f"boolean value: with_N={with_N}, with_M={with_M}, mpc_run={mpc_run}, with_terminal_cost_={with_terminal_cost_}")
    print("PRESS A BUTTON TO CONTINUE")
    input()
    for config_init_state in total_config:
        #FIRST case --> M without terminal cost
        state_buffer,cost_buffer = mpc_double_pendulum.simulation(config_init_state,see_simulation)
        print("finish  M without terminal cost")
        
        #SECOND case --> M + NN as terminal cost
        state_buffer,cost_buffer = mpc_double_pendulum.simulation(config_init_state,see_simulation)
        print("finish  M + NN as terminal cost")
        
        #THIRD case --> N + M without terminal cost
        state_buffer,cost_buffer = mpc_double_pendulum.simulation(config_init_state,see_simulation)
        print("finish  N + M without terminal cost")
        
        #--------------------------SPECIAL CASE:-------------------------------
        
        #FOURTH case --> M + NN as terminal cost
        state_buffer,cost_buffer = mpc_double_pendulum.simulation(config_init_state,see_simulation)
        print("finish M + NN as terminal cost")
        
        #FIVETH case --> M + classic terminal cost
        state_buffer,cost_buffer = mpc_double_pendulum.simulation(config_init_state,see_simulation)
        print("finish M + classic terminal cost")
        
        #SIXTH case -->  M + hybrid cost
        state_buffer,cost_buffer = mpc_double_pendulum.simulation(config_init_state,see_simulation)
        print("finish M + hybrid cost")
        
    print("Total script time:", clock() - time_start)