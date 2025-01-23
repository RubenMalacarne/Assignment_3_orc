from mpc_double_pendulum import DoublePendulumMPC
from conf_double_pendulum import *
from ocp_double_pendulum import *
import re

def animate_double_pendulum(file_path, key="q_trajectory"):
    data = np.load(file_path, allow_pickle=True)

    if key not in data:
        print(f"key '{key}'not fount on the file '{file_path}'.")
        return

    q_trajectory = data[key]  
    q_trajectory = np.vstack(q_trajectory)
    L1 = config.L1
    L2 = config.L2

    fig, ax = plt.subplots()
    ax.set_xlim(-L1 - L2 - 0.1, L1 + L2 + 0.1)
    ax.set_ylim(-L1 - L2 - 0.1, L1 + L2 + 0.1)
    line, = ax.plot([], [], 'o-', lw=2)

    def init():
        line.set_data([], [])
        return line,

    def update(frame):
        q1 = q_trajectory[frame, 0]  
        q2 = q_trajectory[frame, 1] 

        x1 = L1 * np.sin(q1)
        y1 = L1 * np.cos(q1) 
        x2 = x1 + L2 * np.sin(q1 + q2)
        y2 = y1 + L2 * np.cos(q1 + q2)

        line.set_data([0, x1, x2], [0, y1, y2])
        return line,

    ani = FuncAnimation(fig, update, frames=len(q_trajectory), init_func=init, blit=True)
    plt.show()
    
def print_trajectory_from_file(file_path, key="q_trajectory"):
    try:
        data = np.load(file_path, allow_pickle=True)
        
        if key in data:
            q_total_trajectory = data[key]
            q_total_trajectory = np.vstack(q_total_trajectory)
            q1 = q_total_trajectory[:, 0] 
            q2 = q_total_trajectory[:, 1] 

            plt.figure(figsize=(10, 6))
            plt.plot(q1, label='q1', linestyle='-', marker='o')
            plt.plot(q2, label='q2', linestyle='-', marker='x')

            plt.xlabel('Step')
            plt.ylabel('Values')
            plt.title('Trajectory of q1 and q2')
            plt.legend()
            plt.grid()
            plt.show()
        else:
            print(f"key '{key}' not found on the file.")

        data.close()
    except Exception as e:
        print(f"Error, the file is wrong!!!: {e}")

def animate_plots_together(file_paths, key="q_trajectory"):
    num_files = len(file_paths)
    cols = 2
    rows = (num_files + 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(12, 8))
    axes = axes.flatten() 

    lines_q1 = []
    lines_q2 = []
    trajectories = []

    for i, file_path in enumerate(file_paths):
        try:
            match = re.search(r"(mpc_\S+)", file_path)
            if match:
                title_str = match.group(1).replace(".npz","")
            else:
                title_str = "No MPC title found"
            
            data = np.load(file_path, allow_pickle=True)
            if key in data:
                q_trajectory = np.vstack(data[key])
                trajectories.append(q_trajectory)

                ax = axes[i]
                ax.set_xlim(0, len(q_trajectory)) 
                ax.set_ylim(-2 * np.pi, 2 * np.pi)
                ax.set_title(f"{title_str}")
                ax.grid()

                line_q1, = ax.plot([], [], 'r-', lw=2, label="q1")
                line_q2, = ax.plot([], [], 'b-', lw=2, label="q2")
                lines_q1.append(line_q1)
                lines_q2.append(line_q2)

                ax.legend()
            else:
                axes[i].text(0.5, 0.5, f"key '{key}' missing", ha='center', va='center')
        except Exception as e:
            axes[i].text(0.5, 0.5, f"error: {e}", ha='center', va='center')

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    def init():
        for line1, line2 in zip(lines_q1, lines_q2):
            line1.set_data([], [])
            line2.set_data([], [])
        return lines_q1 + lines_q2

    def update(frame):
        for idx, q_trajectory in enumerate(trajectories):
            if frame < len(q_trajectory):
                q1 = q_trajectory[:frame + 1, 0]
                q2 = q_trajectory[:frame + 1, 1]
                lines_q1[idx].set_data(range(len(q1)), q1)
                lines_q2[idx].set_data(range(len(q2)), q2)
        return lines_q1 + lines_q2

    max_frames = max(len(traj) for traj in trajectories)
    ani = FuncAnimation(fig, update, frames=max_frames, init_func=init, blit=True)

    plt.tight_layout()
    plt.show()
    
def plot_all_trajectories(file_paths, key="q_trajectory"):
    num_files = len(file_paths)
    cols = 2
    rows = (num_files + 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(12, 8))
    axes = axes.flatten()  
    for i, file_path in enumerate(file_paths):
        try:
            match = re.search(r"(mpc_\S+)", file_path)
            if match:
                title_str = match.group(1).replace(".npz","")
            else:
                title_str = "No MPC title found"
            data = np.load(file_path, allow_pickle=True)
            if key in data:
                q_total_trajectory = np.vstack(data[key])
                q1 = q_total_trajectory[:, 0]
                q2 = q_total_trajectory[:, 1]

                axes[i].plot(q1, label="q1", linestyle="-", marker="o", markersize=4)
                axes[i].plot(q2, label="q2", linestyle="-", marker="x", markersize=4)
                axes[i].set_title(f"Trajectory {title_str}")
                axes[i].set_xlabel("Step")
                axes[i].set_ylabel("Values")
                axes[i].legend()
                axes[i].grid()
            else:
                axes[i].text(0.5, 0.5, f"key '{key}' missing", ha='center', va='center')
        except Exception as e:
            axes[i].text(0.5, 0.5, f"Error: {e}", ha='center', va='center')
    
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()

def animate_all_simulations_together(file_paths, key="q_trajectory"):
    num_files = len(file_paths)
    cols = 2
    rows = (num_files + 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(12, 8))
    axes = axes.flatten()  

    lines = []
    trajectories = []
    configs = []
    trajectory_lines = []  
    
    for i, file_path in enumerate(file_paths):
        try:
            match = re.search(r"(mpc_\S+)", file_path)
            if match:
                title_str = match.group(1).replace(".npz","")
            else:
                title_str = "No MPC title found"
            data = np.load(file_path, allow_pickle=True)
            if key in data:
                q_trajectory = np.vstack(data[key])
                trajectories.append(q_trajectory)
                L1 = config.L1
                L2 = config.L2

                ax = axes[i]
                ax.set_xlim(-L1 - L2 - 0.1, L1 + L2 + 0.1)
                ax.set_ylim(-L1 - L2 - 0.1, L1 + L2 + 0.1)
                ax.set_title(f"Animation {title_str}")
                ax.grid()

                line, = ax.plot([], [], 'o-', lw=2)

                traj_line_1, = ax.plot([], [], 'r--', lw=1, label='Trajectory Link 1')
                traj_line_2, = ax.plot([], [], 'b--', lw=1, label='Trajectory Link 2')

                lines.append(line)
                trajectory_lines.append((traj_line_1, traj_line_2))
                configs.append((L1, L2))
            else:
                axes[i].text(0.5, 0.5, f"key '{key}' missing", ha='center', va='center')
        except Exception as e:
            axes[i].text(0.5, 0.5, f"Error: {e}", ha='center', va='center')

    for j in range(len(trajectories), len(axes)):
        axes[j].axis("off")

    def init():
        for line, (traj_line_1, traj_line_2) in zip(lines, trajectory_lines):
            line.set_data([], [])
            traj_line_1.set_data([], [])
            traj_line_2.set_data([], [])
        return lines + [t for pair in trajectory_lines for t in pair]

    def update(frame):
        for idx, q_trajectory in enumerate(trajectories):
            if frame >= len(q_trajectory): 
                continue
            q1 = q_trajectory[:frame + 1, 0]
            q2 = q_trajectory[:frame + 1, 1]
            L1, L2 = configs[idx]

            x1 = L1 * np.sin(q_trajectory[frame, 0])
            y1 = L1 * np.cos(q_trajectory[frame, 0])
            x2 = x1 + L2 * np.sin(q_trajectory[frame, 0] + q_trajectory[frame, 1])
            y2 = y1 + L2 * np.cos(q_trajectory[frame, 0] + q_trajectory[frame, 1])

            lines[idx].set_data([0, x1, x2], [0, y1, y2])

            traj_x1 = L1 * np.sin(q_trajectory[:frame + 1, 0])
            traj_y1 = L1 * np.cos(q_trajectory[:frame + 1, 0])
            traj_x2 = traj_x1 + L2 * np.sin(q_trajectory[:frame + 1, 0] + q_trajectory[:frame + 1, 1])
            traj_y2 = traj_y1 + L2 * np.cos(q_trajectory[:frame + 1, 0] + q_trajectory[:frame + 1, 1])

            trajectory_lines[idx][0].set_data(traj_x1, traj_y1)
            trajectory_lines[idx][1].set_data(traj_x2, traj_y2)

        return lines + [t for pair in trajectory_lines for t in pair]

    max_frames = max(len(traj) for traj in trajectories)
    ani = FuncAnimation(fig, update, frames=max_frames, init_func=init, blit=True)

    plt.tight_layout()
    plt.legend(loc="upper right")
    plt.show()

def all_mpc_time(file_paths,key="t_mpc"):
    t_mpc_values = []
    labels = []

    for i, file_path in enumerate(file_paths):
        try:
            match = re.search(r"(mpc_\S+)", file_path)
            if match:
                title_str = match.group(1).replace(".npz","")
            else:
                title_str = "No MPC title found"
            data = np.load(file_path, allow_pickle=True)
            if key in data:
                t_mpc_values.append(data[key].item())
                labels.append(f"{title_str}")
            else:
                print(f"key '{key}' not found on the file'{file_path}'.")
        except Exception as e:
            print(f"Error during load the file '{file_path}': {e}")

    if not t_mpc_values:
        print(f"not value {key} found.")
        return

    min_value = min(t_mpc_values)
    min_index = t_mpc_values.index(min_value)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].bar(labels, t_mpc_values, color='skyblue', edgecolor='black')
    axes[0].set_title("Time MPC for simulation")
    axes[0].set_ylabel("time (s)")
    axes[0].set_xticklabels(labels, rotation=45)
    sorted_indices = sorted(range(len(t_mpc_values)), key=lambda k: t_mpc_values[k])
    text = "\n".join(
        [
            f"{labels[i]} : {t_mpc_values[i]:.3f} s"
            for i in sorted_indices
        ]
    )
    axes[1].text(0.5, 0.5, text, fontsize=12, ha='center', va='center', wrap=True)
    axes[1].axis("off")
    print(f"BEST MPC to reach the goal: \033[95m*Simulation {min_index + 1}* with {min_value:.3f} s.\033[0m")

    plt.tight_layout()
    plt.show()
    
# ---------------------------------------------------------------------
#          MAIN
# ---------------------------------------------------------------------
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
    with_N  = False
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
    
    
    
    counter_config = 0
    
    for config_init_state in total_config:
        # type of configuration
        counter_config += 1
        
        #FIRST case --> M without terminal cost
        filename_mpc = f'save_results/config_{counter_config}/config_{counter_config}_results_mpc_M.npz'
        mpc_double_pendulum.simulation(config_initial_state = config_init_state,see_simulation=see_simulation)
        mpc_double_pendulum.save_result_mpc(filename_mpc)
        print("finish  M without terminal cost and save result")
        filenpz = "save_results/config_1/config_1_results_mpc_M.npz"
        
        # #SECOND case --> N + M without terminal cost
        filename_mpc = f'save_results/config_{counter_config}/config_{counter_config}_results_mpc_M_N.npz'
        mpc_double_pendulum.simulation(with_N,config_initial_state = config_init_state,see_simulation=see_simulation)
        mpc_double_pendulum.save_result_mpc(filename_mpc)
        print("finish  N + M without terminal cost and save result")
        
        # #THIRD case --> M + classic terminal cost
        filename_mpc = f'save_results/config_{counter_config}/config_{counter_config}_results_mpc_M_terminal_cost_standard.npz'
        mpc_double_pendulum.simulation(config_initial_state = config_init_state,see_simulation=see_simulation)
        mpc_double_pendulum.save_result_mpc(filename_mpc)
        print("finish M + classic terminal cost and save result")
        
        # #FOURTH case --> M + NN as terminal cost
        filename_mpc = f'save_results/config_{counter_config}/config_{counter_config}_results_mpc_M_NN.npz'
        mpc_double_pendulum.simulation(config_initial_state = config_init_state,see_simulation=see_simulation,term_cost_NN_=term_cost_NNet)
        mpc_double_pendulum.save_result_mpc(filename_mpc)
        print("finish  M + NN as terminal cost and save result")
        
        file_paths = [
        "save_results/config_1/config_1_results_mpc_M.npz",
        "save_results/config_1/config_1_results_mpc_M_N.npz",
        "save_results/config_1/config_1_results_mpc_M_terminal_cost_standard.npz",
        "save_results/config_1/config_1_results_mpc_M_NN.npz",
        ]
    file_paths = [
    "save_results/config_1/config_1_results_mpc_M.npz",
    "save_results/config_1/config_1_results_mpc_M_NN.npz",
    "save_results/config_1/config_1_results_mpc_M_N.npz",
    "save_results/config_1/config_1_results_mpc_M_terminal_cost_standard.npz",
    ]
    plot_all_trajectories(file_paths)
    animate_all_simulations_together(file_paths)
    animate_plots_together(file_paths)
    all_mpc_time(file_paths)

    print("Total script time:", clock() - time_start)
  