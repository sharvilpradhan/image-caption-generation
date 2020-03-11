import numpy as np
from data_generator import DataGenerator
from model import Model
from config import Config
import itertools
from nnmodel import *


class ImageSearch:
	def __init__(self):
		self.config = Config()
		self.data_generator = DataGenerator()
		self.model = Model()
		self.all_image_embeddings = []
		self.all_image_filenames = []

		if(config.use_IC_V6):
			self.model = NNModel(self.data_generator.tokens)
			self.inception = Model().inception
		else:
			self.model = Model()
			self.inception = self.model.inception

		self.model.load_state_dict(torch.load(self.config.trained_model))
		Model.set_model_untrainable(self.model)

		for i in range(len(self.data_generator.training_data)):
			self.all_image_embeddings.append(self.inception(self.data_generator.convert_image_to_tensor(self.config.image_dir + self.data_generator.training_data[i]['filename'])).detach().numpy())
			self.all_image_filenames.append(self.data_generator.training_data[i]['filename'])


	def return_cosine_sorted_image(self,target_image_embedding):
		cosines = []		
		for i in range(len(self.all_image_embeddings)):
			cosines.append(1 - spatial.distance.cosine(target_image_embedding, self.all_image_embeddings[i]))
			
		sorted_indexes = np.argsort(cosines)[::-1]
		
		return np.vstack((np.array(self.all_image_filenames)[sorted_indexes], np.array(cosines)[sorted_indexes])).T

	def return_embedding_image(image_filename):
		return self.model.inception(self.data_generator.convert_image_to_tensor(image_filename)).detach().numpy().squeeze()

	def search_image_from_caption(self,caption):
		tokens= self.data_generator.convert_sentence_to_tokens(caption)
		embedding_tensor = torch.autograd.Variable(torch.randn(1,self.config.input_embedding)*0.01, requires_grad=True)
		l = torch.nn.CrossEntropyLoss(reduction='none')
		epochs = 1000
		loss_so_far = 0.0
		lr = 0.001
		with torch.autograd.set_detect_anomaly(True):
			for epoch in range(epochs):
				input_token = self.data_generator.w2i[self.config.start_word]
				input_tensor = torch.tensor(input_token)
				loss=0.

				for token in tokens:
					if(input_token==self.data_generator.w2i[self.config.start_word]):
						out, hidden=net(input_tensor, embedding_tensor, inception = self.inception, process_image=True, use_inception=False)
					else:
						out, hidden=net(input_tensor, hidden)

					class_label = torch.tensor(token).view(1)
					input_token = token
					input_tensor = torch.tensor(input_token)
					out = out.squeeze().view(1,-1)
					loss = loss + l(out,class_label)
				loss.backward()
				embedding_tensor =  torch.autograd.Variable(embedding_tensor.clone() - lr * embedding_tensor.grad, requires_grad=True)
				loss_so_far += loss.detach().item()

				if(epoch %10 ==0):
					print("==== Epoch: ",epoch, " loss: ",round(loss.detach().item(),3)," | running avg loss: ", round(loss_so_far/(epoch+1),3))
					if(epoch %100 == 0):
						similar_images = return_cosine_sorted_image(embedding_tensor.detach().numpy().squeeze())
						print(similar_images[:3])
		return similar_images
				