import torch
import torch.nn as nn
import os


class DeepQNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, hidden_layers):
        super().__init__()
        self.input = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU(inplace=True))
        self.hidden = []
        for i in range(hidden_layers):
            self.hidden.append(nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU(inplace=True)))
        self.output = nn.Linear(hidden_size, output_size)

        # weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.input(x)
        for f in self.hidden:
            x = f(x)
        x = self.output(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        self.loss_func = nn.MSELoss()

    def train_step(self, next_states, slice=True):
        # eval mode
        self.model.eval()
        with torch.no_grad():
            if slice:
                preds = self.model(next_states)[:, 0]
            else:
                preds = self.model(next_states)

        # back to train mode
        self.model.train()

        return preds
