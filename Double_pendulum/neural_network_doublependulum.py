import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from casadi import MX, Function
import l4casadi as l4c
import copy

import conf_double_pendulum as config

class NeuralNetwork(nn.Module):
    """ A simple feedforward neural network that predicts the *log* of the cost (scaled). """
    def __init__(self, file_name,
                 input_size = 4,
                 hidden_size = 12,
                 output_size = 1,
                 n_epochs=100,
                 batch_size=10,
                 lr=0.0001,
                 activation_type=nn.Tanh()):
        super().__init__()

        #load data 
        self.X_train, self.X_test, self.y_train, self.y_test, self.log_min, self.log_max = self.read_file_csv(file_name)
        
        self.X_train_t = torch.tensor(self.X_train, dtype=torch.float32)
        self.y_train_t = torch.tensor(self.y_train, dtype=torch.float32).reshape(-1, 1)
        self.X_test_t  = torch.tensor(self.X_test,  dtype=torch.float32)
        self.y_test_t  = torch.tensor(self.y_test,  dtype=torch.float32).reshape(-1, 1)

        self.input_size    = input_size
        self.hidden_size   = hidden_size
        self.output_size   = output_size
        self.activation    = activation_type
        self.n_epochs      = n_epochs
        self.batch_size    = batch_size

        self.line_stack = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            self.activation,
            nn.Linear(self.hidden_size, self.hidden_size),
            self.activation,
            nn.Linear(self.hidden_size, self.output_size)
        )
        self.initialize_weights()

        self.loss_fn    = nn.MSELoss()
        self.optimizer  = optim.Adam(self.line_stack.parameters(), lr=lr)
        self.history    = []
        # self.ub = 1.0  #ub --> upper bound: used to produce dates with un upper limits, to prevent possible out of range 
        # self.batch_start = torch.arange(0, len(self.X_train_t), self.batch_size)

    def forward(self, x):
        # La rete produce un singolo valore: il "log-cost scalato"
        # (cioè un numero che si trova idealmente in [-1,1], ma può sforare).
        return self.line_stack(x)

    def initialize_weights(self):
        for layer in self.line_stack:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def trainig_part(self):
        print("Training the model on scaled log(cost) ...")
        best_mse = float('inf')
        best_weights = None

        for epoch in range(self.n_epochs):
            self.line_stack.train()

            idx_list = np.arange(len(self.X_train_t))
            np.random.shuffle(idx_list)
            batches = [
                idx_list[i:i+self.batch_size]
                for i in range(0, len(idx_list), self.batch_size)
            ]

            pbar = tqdm.tqdm(batches, desc=f"Epoch {epoch+1}")
            for batch_indices in pbar:
                X_batch = self.X_train_t[batch_indices]
                y_batch = self.y_train_t[batch_indices]

                self.optimizer.zero_grad()
                y_pred = self.line_stack(X_batch)
                loss   = self.loss_fn(y_pred, y_batch)
                loss.backward()
                self.optimizer.step()

            self.line_stack.eval()
            with torch.no_grad():
                y_pred_test = self.line_stack(self.X_test_t)
                mse = self.loss_fn(y_pred_test, self.y_test_t).item()
                self.history.append(mse)
                if mse < best_mse:
                    best_mse = mse
                    best_weights = copy.deepcopy(self.line_stack.state_dict())

        # restore weights
        self.line_stack.load_state_dict(best_weights)
        print(f"Training complete. Best MSE (scaled log-cost): {best_mse:.4f}")
        print("MSE:", best_mse)
        print("RMSE:", np.sqrt(best_mse))

    def create_casadi_function(self, robot_name, NN_DIR, input_size, load_weights=True):
        from casadi import MX, Function
        import casadi as cs
        import l4casadi as l4c
        # if load_weights is True, we load the neural-network weights from a ".pt" file
        if load_weights:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            nn_name = f'{NN_DIR}model.pt'
            nn_data = torch.load(nn_name, map_location=device)
            self.load_state_dict(nn_data['model'])

        state = MX.sym("x", 1, input_size)

        self.l4c_model  = l4c.L4CasADi(
                                    self,
                                    device='cuda' if torch.cuda.is_available() else 'cpu',
                                    name=f'{robot_name}_model',
                                    build_dir=f'{NN_DIR}nn_{robot_name}')
        scaled_log_cost = self.l4c_model(state)  

        #da spiegare
        real_log_cost = ((scaled_log_cost + 1.0)/2.0) * (self.log_max - self.log_min) + self.log_min
        cost_pred     = cs.exp(real_log_cost)  # exponent --> costo sempre > 0
        self.nn_func = Function('nn_func', [state], [cost_pred])

    def read_file_csv(self, file_name):
        df = pd.read_csv(file_name)
        X_data  = df[["q1","q2","v1","v2"]].values  # shape (N,4)
        costs   = df["cost"].values                # shape (N,)
        log_cost = np.log(costs)
        self.scaler = MinMaxScaler(feature_range=(-1, 1))

        log_cost_scaled = self.scaler.fit_transform(log_cost.reshape(-1,1)).flatten()
        log_min = float(self.scaler.data_min_)
        log_max = float(self.scaler.data_max_)

        X_train, X_test, y_train, y_test = train_test_split(
                                                            X_data, log_cost_scaled,
                                                            train_size=0.7,
                                                            shuffle=True
                                                        )

        print("Shape dataset X:", X_data.shape, " log_cost:", log_cost_scaled.shape)
        print("Train shapes:", X_train.shape, y_train.shape,
              " Test shapes:", X_test.shape, y_test.shape)

        return X_train, X_test, y_train, y_test, log_min, log_max

    def plot_training_history(self):
        plt.figure()
        plt.plot(self.history, '-o')
        plt.xlabel("Epoch")
        plt.ylabel("MSE on test (scaled log-cost)")
        plt.title("Training History")
        plt.grid(True)
        plt.show()


# ---------------------------------------------------------------------
#          MAIN(example)
# ---------------------------------------------------------------------
    
if __name__ == "__main__":
    file_name = "models/ocp_dataset_DP_train.csv"
    net = NeuralNetwork(file_name)
    net.trainig_part()
    net.plot_training_history()

    torch.save({'model': net.state_dict()}, "models/model.pt")
    print("Model saved.")
