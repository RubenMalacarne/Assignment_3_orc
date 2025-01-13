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
    """ A simple feedforward neural network. """
    def __init__(self, file_name,input_size = 4, hidden_size = 12, output_size = 1,n_epochs=100,batch_size=10, lr=0.0001 ,activation_type=nn.Tanh(), ub=None):
        super().__init__()
        
        self.X_train, self.X_test, self.y_train, self.y_test = self.read_file_csv(file_name)
        self.X_train_t = torch.tensor(self.X_train, dtype=torch.float32)
        self.y_train_t = torch.tensor(self.y_train, dtype=torch.float32).reshape(-1, 1)
        self.X_test_t = torch.tensor(self.X_test, dtype=torch.float32)
        self.y_test_t = torch.tensor(self.y_test, dtype=torch.float32).reshape(-1, 1)
        
        self.input_size = input_size
        self.hidden_size =hidden_size
        self.output_size =output_size
        self.activation_type=activation_type
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.ub = 1.0  
        #define model
        self.model = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            self.activation_type,
            nn.Linear(self.hidden_size, self.hidden_size),
            self.activation_type,
            nn.Linear(self.hidden_size, self.output_size)
        )
        
        self.loss_fn = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        self.history = []
        self.batch_start = torch.arange(0, len(self.X_train_t), self.batch_size)

   
    def read_file_csv(self,file_name):
        
        df = pd.read_csv(file_name)
        X = df[["q1", "v1","q2", "v2"]].values
        #rescale_between [-1 and 1]
        y = df["cost"].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(-1, 1))
        y = scaler.fit_transform(y).flatten()
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True)
        print(np.shape(X), np.shape(y))
        print(np.shape(X_train), np.shape(y_train))
        print(np.shape(X_test), np.shape(y_test))
        return X_train, X_test, y_train, y_test
    

    def trainig_part(self):
        
        print("Training the model ...")
        best_mse = float('inf')
        best_weights = None
        for epoch in range(self.n_epochs):
            self.model.train()
            for start in tqdm.tqdm(range(0, len(self.X_train_t), self.batch_size), desc=f"Epoch {epoch+1}"):
                X_batch = self.X_train_t[start:start+self.batch_size]
                y_batch = self.y_train_t[start:start+self.batch_size]

                self.optimizer.zero_grad()
                y_pred = self.model(X_batch)
                loss = self.loss_fn(y_pred, y_batch)
                loss.backward()
                self.optimizer.step()

            # Validate at the end of the epoch
            self.model.eval()
            with torch.no_grad():
                y_pred = self.model(self.X_test_t)
                mse = self.loss_fn(y_pred, self.y_test_t).item()
                self.history.append(mse)
                if mse < best_mse:
                    best_mse = mse
                    best_weights = self.model.state_dict()
        
        # restore model and return best accuracy
        self.model.load_state_dict(best_weights)
        print(f"Training complete. Best MSE: {best_mse:.4f}")
        
        print("MSE: %.2f" % best_mse)
        print("RMSE: %.2f" % np.sqrt(best_mse))

    def forward(self, x):
        out = self.model(x) * self.ub
        return out
   
    def initialize_weights(self):
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias) 

    def create_casadi_function(self, robot_name, input_size, load_weights):
        # if load_weights is True, we load the neural-network weights from a ".pt" file
        if(load_weights):
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            nn_name = f'model.pt'
            nn_data = torch.load(nn_name, map_location=device,weights_only=True)
            self.load_state_dict(nn_data)

        state = MX.sym("x", input_size)        
        self.l4c_model = l4c.L4CasADi(self,
                                      device='cuda' if torch.cuda.is_available() else 'cpu',
                                      name=f'{robot_name}_model',
                                      build_dir=f'nn_{robot_name}')
        self.nn_model = self.l4c_model(state)
        # This is the function that you can use in a casadi problem
        self.nn_func = Function('nn_func', [state], [self.nn_model])
    
    def plot_training_history(self):
        plt.plot(self.history)
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.title("Training Loss")
        plt.show()
        
        
if __name__ == "__main__":
    nn = NeuralNetwork(file_name="ocp_dataset_DP_train.csv")
    nn.trainig_part()
    nn.plot_training_history()

    # Save the trained model
    torch.save( nn.model.state_dict(), "model.pt")
