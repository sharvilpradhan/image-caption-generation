from data_generator import DataGenerator
from nnmodel import *
from model import Model
from config import Config
import torch

class Inference:
	def __init__(self):
		self.data_generator = DataGenerator()
		self.config = Config()

		if(config.use_IC_V6):
			self.model = NNModel(self.data_generator.tokens)
			self.inception = Model().inception
		else:
			self.model = Model()
			self.inception = self.model.inception

		self.model.load_state_dict(torch.load(self.config.trained_model))

		if(self.config.use_gpu):
			self.model.cuda()
			self.inception.cuda()
		self.model.eval()
		self.inception.eval()


	def get_caption(self,image_filename):
		return self.data_generator.get_caption(self.model, self.inception, image_filename)
