import torch
import torch.nn as nn

class Discriminator(nn.Module):
	def __init__(
		self,
		seq_len=60,
		n_feat=4,
		hidden_dim=64, 
		n_layers=4,
		**kwargs
	):
		super().__init__()
		self.seq_len = seq_len
		self.n_feat = n_feat
		self.dropout = kwargs.get('dropout') if kwargs.get('dropout') else 0
		self.loss_func = torch.nn.BCELoss()

		self.lstm = nn.LSTM(n_feat, hidden_dim, n_layers, dropout=self.dropout)

		self.mlp = nn.Sequential(
			nn.Linear(hidden_dim, hidden_dim * 2),
			nn.LeakyReLU(0.1),
			nn.Linear(hidden_dim * 2, n_feat),
			nn.Tanh()
		)

		self.flatten = nn.Flatten()
		self.output = nn.Sequential(
			nn.Linear(seq_len * n_feat, 1),
			nn.Sigmoid()
		)
		
	def forward(self, x):
		lstm_out, _ = self.lstm(x)
		# print(f'lstm output shape: {lstm_out.shape}')
		mlp_out = self.mlp(lstm_out)
		# print(f'output shape: {output.shape}')
		output = self.flatten(mlp_out)
		# print(f'flatten shape: {output.shape}')
		output = self.output(output)
		return output.squeeze()

	def loss(self, output, label):
		return self.loss_func(output, label)