import conf_triple_pendulum as config  # Assicurati che il file di configurazione includa L1, L2, L3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import re

def animate_triple_pendulum(file_path, key="q_trajectory"):
    data = np.load(file_path, allow_pickle=True)

    if key not in data:
        print(f"Key '{key}' non trovata nel file '{file_path}'.")
        return

    q_trajectory = data[key]  
    q_trajectory = np.vstack(q_trajectory)
    L1, L2, L3 = config.L1, config.L2, config.L3

    fig, ax = plt.subplots()
    max_len = L1 + L2 + L3
    ax.set_xlim(-max_len - 0.1, max_len + 0.1)
    ax.set_ylim(-max_len - 0.1, max_len + 0.1)
    line, = ax.plot([], [], 'o-', lw=2)

    def init():
        line.set_data([], [])
        return line,

    def update(frame):
        q1 = q_trajectory[frame, 0]
        q2 = q_trajectory[frame, 1]
        q3 = q_trajectory[frame, 2]

        x1 = L1 * np.sin(q1)
        y1 = L1 * np.cos(q1)
        x2 = x1 + L2 * np.sin(q1 + q2)
        y2 = y1 + L2 * np.cos(q1 + q2)
        x3 = x2 + L3 * np.sin(q1 + q2 + q3)
        y3 = y2 + L3 * np.cos(q1 + q2 + q3)

        line.set_data([0, x1, x2, x3], [0, y1, y2, y3])
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
            q3 = q_total_trajectory[:, 2]

            plt.figure(figsize=(10, 6), num="Traiettorie q1, q2, q3")
            plt.plot(q1, label='q1', linestyle='-', marker='o', markersize=4)
            plt.plot(q2, label='q2', linestyle='-', marker='x', markersize=4)
            plt.plot(q3, label='q3', linestyle='-', marker='^', markersize=4)

            plt.xlabel('Step')
            plt.ylabel('Valori')
            plt.title('Traiettorie di q1, q2, q3')
            plt.legend()
            plt.grid()
            plt.show()
        else:
            print(f"Key '{key}' non trovata.")

        data.close()
    except Exception as e:
        print(f"Errore: {e}")

def animate_plots_together(file_paths, key="q_trajectory"):
    num_files = len(file_paths)
    cols = 2
    rows = (num_files + 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(12, 8), num="Traiettorie Multiple")
    axes = axes.flatten()

    lines_q1, lines_q2, lines_q3 = [], [], []
    trajectories = []

    for i, file_path in enumerate(file_paths):
        try:
            match = re.search(r"(mpc_\S+)", file_path)
            title_str = match.group(1).replace(".npz", "") if match else "Simulazione"
            
            data = np.load(file_path, allow_pickle=True)
            if key in data:
                q_trajectory = np.vstack(data[key])
                trajectories.append(q_trajectory)

                ax = axes[i]
                ax.set_xlim(0, len(q_trajectory))
                ax.set_ylim(-2 * np.pi, 2 * np.pi)
                ax.set_title(title_str)
                ax.grid()

                line_q1, = ax.plot([], [], 'r-', label="q1")
                line_q2, = ax.plot([], [], 'b-', label="q2")
                line_q3, = ax.plot([], [], 'g-', label="q3")
                lines_q1.append(line_q1)
                lines_q2.append(line_q2)
                lines_q3.append(line_q3)
                ax.legend()
            else:
                axes[i].text(0.5, 0.5, f"Key '{key}' mancante", ha='center', va='center')
        except Exception as e:
            axes[i].text(0.5, 0.5, f"Errore: {e}", ha='center', va='center')

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    def init():
        for line in lines_q1 + lines_q2 + lines_q3:
            line.set_data([], [])
        return lines_q1 + lines_q2 + lines_q3

    def update(frame):
        for idx, traj in enumerate(trajectories):
            if frame < len(traj):
                lines_q1[idx].set_data(range(frame+1), traj[:frame+1, 0])
                lines_q2[idx].set_data(range(frame+1), traj[:frame+1, 1])
                lines_q3[idx].set_data(range(frame+1), traj[:frame+1, 2])
        return lines_q1 + lines_q2 + lines_q3

    max_frames = max(len(t) for t in trajectories)
    ani = FuncAnimation(fig, update, frames=max_frames, init_func=init, blit=True)
    plt.tight_layout()
    plt.show()

def animate_all_simulations_together(file_paths, key="q_trajectory"):
    num_files = len(file_paths)
    cols = 2
    rows = (num_files + 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(12, 8), num="Animazioni Multiple")
    axes = axes.flatten()

    lines, trajectory_lines = [], []
    trajectories, configs = [], []

    for i, file_path in enumerate(file_paths):
        try:
            match = re.search(r"(mpc_\S+)", file_path)
            title_str = match.group(1).replace(".npz", "") if match else "Simulazione"
            
            data = np.load(file_path, allow_pickle=True)
            if key in data:
                q_trajectory = np.vstack(data[key])
                trajectories.append(q_trajectory)
                L1, L2, L3 = config.L1, config.L2, config.L3

                ax = axes[i]
                max_len = L1 + L2 + L3
                ax.set_xlim(-max_len - 0.1, max_len + 0.1)
                ax.set_ylim(-max_len - 0.1, max_len + 0.1)
                ax.set_title(title_str)
                ax.grid()

                line, = ax.plot([], [], 'o-', lw=2)
                traj_line1, = ax.plot([], [], 'r--', lw=1, label='Traiettoria 1')
                traj_line2, = ax.plot([], [], 'b--', lw=1, label='Traiettoria 2')
                traj_line3, = ax.plot([], [], 'g--', lw=1, label='Traiettoria 3')

                lines.append(line)
                trajectory_lines.append((traj_line1, traj_line2, traj_line3))
                configs.append((L1, L2, L3))
            else:
                axes[i].text(0.5, 0.5, f"Key '{key}' mancante", ha='center', va='center')
        except Exception as e:
            axes[i].text(0.5, 0.5, f"Errore: {e}", ha='center', va='center')

    for j in range(len(trajectories), len(axes)):
        axes[j].axis("off")

    def init():
        for line in lines:
            line.set_data([], [])
        for traj_lines in trajectory_lines:
            for tl in traj_lines:
                tl.set_data([], [])
        return lines + [tl for traj in trajectory_lines for tl in traj]

    def update(frame):
        for idx, (traj, (L1, L2, L3)) in enumerate(zip(trajectories, configs)):
            if frame >= len(traj):
                continue
            q1, q2, q3 = traj[frame, :3]
            
            # Calcola posizioni correnti
            x1 = L1 * np.sin(q1)
            y1 = L1 * np.cos(q1)
            x2 = x1 + L2 * np.sin(q1 + q2)
            y2 = y1 + L2 * np.cos(q1 + q2)
            x3 = x2 + L3 * np.sin(q1 + q2 + q3)
            y3 = y2 + L3 * np.cos(q1 + q2 + q3)
            lines[idx].set_data([0, x1, x2, x3], [0, y1, y2, y3])

            # Traiettorie cumulative
            traj_x1 = L1 * np.sin(traj[:frame+1, 0])
            traj_y1 = L1 * np.cos(traj[:frame+1, 0])
            traj_x2 = traj_x1 + L2 * np.sin(traj[:frame+1, 0] + traj[:frame+1, 1])
            traj_y2 = traj_y1 + L2 * np.cos(traj[:frame+1, 0] + traj[:frame+1, 1])
            traj_x3 = traj_x2 + L3 * np.sin(traj[:frame+1, 0] + traj[:frame+1, 1] + traj[:frame+1, 2])
            traj_y3 = traj_y2 + L3 * np.cos(traj[:frame+1, 0] + traj[:frame+1, 1] + traj[:frame+1, 2])

            trajectory_lines[idx][0].set_data(traj_x1, traj_y1)
            trajectory_lines[idx][1].set_data(traj_x2, traj_y2)
            trajectory_lines[idx][2].set_data(traj_x3, traj_y3)

        return lines + [tl for traj in trajectory_lines for tl in traj]

    max_frames = max(len(t) for t in trajectories)
    ani = FuncAnimation(fig, update, frames=max_frames, init_func=init, blit=True)
    plt.tight_layout()
    plt.legend()
    plt.show()

def plot_all_trajectories(file_paths, key="q_trajectory"):
    num_files = len(file_paths)
    cols = 2
    rows = (num_files + 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 8),num="plot_all_trajectories")
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
                q3 = q_total_trajectory[:, 2]

                axes[i].plot(q1, label="q1", linestyle="-", marker="o", markersize=4)
                axes[i].plot(q2, label="q2", linestyle="-", marker="x", markersize=4)
                axes[i].plot(q3, label="q3", linestyle="-", marker="x", markersize=4)
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


def all_mpc_time(file_paths, keys=["t_mpc", "tot_iteration"]):
    t_mpc_values = []
    tot_iteration_values = []
    labels = []

    for i, file_path in enumerate(file_paths):
        try:
            match = re.search(r"(mpc_\S+)", file_path)
            if match:
                title_str = match.group(1).replace(".npz", "")
            else:
                title_str = "No MPC title found"
            
            data = np.load(file_path, allow_pickle=True)
            
            if keys[0] in data and keys[1] in data:
                t_mpc_values.append(data[keys[0]].item())
                tot_iteration_values.append(data[keys[1]].item())
                labels.append(f"{title_str}")
            else:
                print(f"Keys '{keys}' not found in the file '{file_path}'.")
        except Exception as e:
            print(f"Error during load of the file '{file_path}': {e}")

    if not t_mpc_values or not tot_iteration_values:
        print(f"Values for keys {keys} not found.")
        return

    sorted_indices_t_mpc = sorted(range(len(t_mpc_values)), key=lambda k: t_mpc_values[k])
    t_mpc_ranking = [f"{labels[i]}: {t_mpc_values[i]:.3f} s" for i in sorted_indices_t_mpc]

    sorted_indices_iterations = sorted(range(len(tot_iteration_values)), key=lambda k: tot_iteration_values[k])
    iteration_ranking = [f"{labels[i]}: {tot_iteration_values[i]} iterations" for i in sorted_indices_iterations]

    fig, axes = plt.subplots(2, 2, figsize=(8, 9),num="all_mpc_time")

    axes[0, 0].bar(labels, t_mpc_values, color='skyblue', edgecolor='black')
    axes[0, 0].set_title("Time MPC for simulation")
    axes[0, 0].set_ylabel("time (s)")
    axes[0, 0].set_xticklabels(labels, rotation=45)

    axes[0, 1].text(0.5, 0.5, "\n".join(t_mpc_ranking), fontsize=12, ha='center', va='center', wrap=True)
    axes[0, 1].axis("off")
    axes[0, 1].set_title("Sorted t_mpc values")

    axes[1, 0].bar(labels, tot_iteration_values, color='lightgreen', edgecolor='black')
    axes[1, 0].set_title("Total Iterations for simulation")
    axes[1, 0].set_ylabel("iterations")
    axes[1, 0].set_xticklabels(labels, rotation=45)

    axes[1, 1].text(0.5, 0.5, "\n".join(iteration_ranking), fontsize=12, ha='center', va='center', wrap=True)
    axes[1, 1].axis("off")
    axes[1, 1].set_title("Sorted tot_iteration values")

    min_time_value = min(t_mpc_values)
    min_time_index = t_mpc_values.index(min_time_value)

    print(f"BEST MPC to reach the goal: \033[95m*Simulation {min_time_index + 1}* with {min_time_value:.3f} s.\033[0m")
    print("Ranking by t_mpc:")
    print("\n".join(t_mpc_ranking))
    print("Ranking by iterations:")
    print("\n".join(iteration_ranking))

    plt.tight_layout()
    plt.show()

def plot_joint_dynamics(file_path, keys=["q_trajectory", "dq_total", "ddq_total"]):
    try:
        data = np.load(file_path, allow_pickle=True)
        trajectories = [np.vstack(data[key]) if key in data else None for key in keys]

        if any(traj is None for traj in trajectories):
            print(f"one or more key: {keys} are not present in the file.")
            return

        q_trajectory, dq_trajectory, ddq_trajectory = trajectories

        fig, axs = plt.subplots(3, 1, figsize=(10, 12),num="plot_joint_dynamics")
        titles = ["Joint Positions", "Joint Velocities", "Joint Accelerations"]
        y_labels = ["Position (rad)", "Velocity (rad/s)", "Acceleration (rad/s²)"]

        for ax, traj, title, ylabel in zip(axs, trajectories, titles, y_labels):
            for joint_idx in range(traj.shape[1]):
                ax.plot(traj[:, joint_idx], label=f"Joint {joint_idx + 1}")

            ax.set_title(title)
            ax.set_xlabel("Iteration")
            ax.set_ylabel(ylabel)
            ax.legend()
            ax.grid()

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error during loading the plots: {e}")


def plot_joint_acceleration(file_paths, key="ddq_total"):
    num_files = len(file_paths)
    cols = 2
    rows = (num_files + 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 8), num="plot_joint_acceleration")
    axes = axes.flatten()
    
    for i, file_path in enumerate(file_paths):
        try:
            match = re.search(r"(mpc_\S+)", file_path)
            if match:
                title_str = match.group(1).replace(".npz", "")
            else:
                title_str = "No MPC title found"
            
            data = np.load(file_path, allow_pickle=True)
            if key in data:
                ddq_total_trajectory = np.vstack(data[key])
                
                for joint_idx in range(ddq_total_trajectory.shape[1]):
                    axes[i].plot(ddq_total_trajectory[:, joint_idx], label=f"Joint {joint_idx + 1}", linestyle="-", marker="o", markersize=4)
                
                axes[i].set_title(f"Acceleration {title_str}")
                axes[i].set_xlabel("Step")
                axes[i].set_ylabel("Acceleration (rad/s²)")
                axes[i].legend()
                axes[i].grid()
            else:
                axes[i].text(0.5, 0.5, f"Key '{key}' missing", ha='center', va='center')
        except Exception as e:
            axes[i].text(0.5, 0.5, f"Error: {e}", ha='center', va='center')
    
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    
    plt.tight_layout()
    plt.show()