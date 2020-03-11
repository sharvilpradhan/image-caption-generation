from config import Config
from data_generator import DataGenerator
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim  
import torchvision.transforms.functional as TF

import torchvision
from torchvision import datasets, models, transforms

class Model(nn.Module):
	def __init__(self):
		super(Model, self).__init__()
		self.config = Config()
		self.load_inception_v3()
		self.data_generator = DataGenerator()
		self.construct_model()

	def load_inception_v3(self):
		self.inception = models.inception_v3(pretrained=True)
		Model.set_model_untrainable(self.inception)
		self.num_features = self.inception.fc.in_features 
		self.inception.fc = nn.Linear(self.num_features,self.config.input_embedding)
		self.inception.load_state_dict(torch.load(self.config.inception_model))

		if(self.config.use_gpu):
			self.inception.cuda()

	def set_model_untrainable(model):
		for param in model.parameters():
			param.requires_grad = False

	def construct_model(self):
		self.batch_norm = nn.BatchNorm1d(self.config.input_embedding)
		self.input_embedding = nn.Embedding(self.data_generator.tokens, self.config.input_embedding)
		self.embedding_dropout = nn.Dropout(p = self.config.dropout)
		self.gru = nn.GRU(input_size=self.config.input_embedding, hidden_size=self.config.hidden_size, num_layers=self.config.gru_layers, dropout=self.config.dropout)
		self.linear = nn.Linear(self.config.hidden_size, self.config.output_embedding)
		self.out = nn.Linear(self.config.output_embedding, self.data_generator.tokens)

	def forward(self, input_tokens, hidden, process_image=False, use_inception=True):
		device = torch.device('cuda' if self.config.use_gpu else 'cpu')
		if(process_image):
			inp=self.embedding_dropout(self.inception(hidden)) if use_inception else hidden
			hidden=torch.zeros((self.gru_layers,1, self.hidden_state_size))
		else:
			inp=self.embedding_dropout(self.input_embedding(input_tokens.view(1).type(torch.LongTensor).to(device)))			
		
		hidden = hidden.view(self.config.gru_layers,1,-1)
		inp = inp.view(1,1,-1)
		out, hidden = self.gru(inp, hidden)
		out = self.out(self.linear(out))
		return out, hidden

	def train_network(self, epochs = 20):
		l = torch.nn.CrossEntropyLoss(reduction='none')
		opt = optim.Adam(self.parameters(), lr=0.0001)
		self.inception.eval()
		self.train()
		loss_so_far = 0.0
		total_samples = len(self.data_generator.training_data)
		for epoch in range(epochs):
			for (image_tensor, tokens, _, index) in self.data_generator.get_train_batch():
				opt.zero_grad()
				self.zero_grad()
				words = []
				loss=0.
				input_token = self.data_generator.w2i[self.config.start_word]
				input_tensor = torch.tensor(input_token)
				for token in tokens:
					if(input_token==self.data_generator.w2i[self.config.start_word]):
						out, hidden=self(input_tensor, image_tensor, process_image=True)
					else:
						out, hidden=self(input_tensor, hidden)
					class_label = torch.tensor(token).view(1)
					input_token = token
					input_tensor = torch.tensor(input_token)
					out = out.squeeze().view(1,-1)
					loss += l(out,class_label)
				loss = loss/len(tokens)
				loss.backward()
				opt.step()
				loss_so_far += loss.detach().item()

	def save_the_network(self):
		torch.save(self.state_dict(), self.config.trained_model)
		torch.save(self.inception.state_dict(), self.config.inception_model)

