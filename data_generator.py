import json
from config import Config
from PIL import Image
import torchvision.transforms.functional as tf
import torch.nn.functional as F
from copy import deepcopy
import numpy as np
import torch

class DataGenerator:
	def __init__(self):
		self.config = Config()
		self.captions_dataset = json.load(open(self.config.captions_file,'r'))['images']

		self.training_data = []
		self.test_data = []
		self.w2i = {self.config.end_word:0, self.config.start_word: 1}
		self.word_frequency = {self.config.end_word:0, self.config.start_word: 0}
		self.i2w = {0: self.config.end_word, 1: self.config.start_word}
		self.tokens = 2
		self.batch_index = 0

		self.parse_caption_data();

	def parse_caption_data(self):
		for file in self.captions_dataset:
			if(file['split'] == 'train'):
				self.training_data.append(file)
			else:
				self.test_data.append(file)

			for sentence in file['sentences']:
				for token in sentence['tokens']:
					if(token not in self.w2i.keys()):
						self.w2i[token] = self.tokens
						self.i2w[self.tokens] = token
						self.tokens +=1
						self.word_frequency[token] = 1
					else:
						self.word_frequency[token] += 1


	def get_train_batch(self):
		for i in range(len(self.training_data)):
			file = self.training_data[i]
			output_sentence_tokens = deepcopy(file['sentences'][np.random.randint(len(file['sentences']))]['tokens'])
			output_sentence_tokens.append(self.config.end_word)
			image = self.convert_image_to_tensor(self.config.image_dir + ('' if self.config.image_dir[-1] == '/' else '/') +  file['filename'])
			yield image, list(map(lambda x: self.w2i[x], output_sentence_tokens)), output_sentence_tokens, i

	def convert_sentence_to_tokens(self,sentence):
		tokens = sentence.split(" ")
		converted_tokens= list(map(lambda x: self.w2i[x], tokens))
		converted_tokens.append(self.w2i[config.end_word])
		return converted_tokens

	def convert_tensor_to_word(self,tensor):
		output = torch.nn.functional.log_softmax(tensor.detach().squeeze(), dim=0).numpy()
		return self.i2w[np.argmax(output)]

	def convert_image_to_tensor(self,filename):
		return torch.unsqueeze(tf.normalize(tf.to_tensor(pic = tf.resize(img=Image.open(filename), size=(self.config.height,self.config.width))), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),0)

	def get_caption(self,model, inception, image_filename, max_words=15):
		image_tensor = self.convert_image_to_tensor(image_filename)
		hidden=None
		embedding=None
		words = []
	
		input_token = self.config.start_word
		input_tensor = torch.tensor(self.w2i[input_token]).type(torch.LongTensor)

		for i in range(max_words):
			out, hidden = model(input_tensor, inception = inception, hidden=image_tensor, process_image=True) if(i == 0) else model(input_tensor, hidden)
			word = self.convert_tensor_to_word(out)
			input_token = self.w2i[word]
			input_tensor = torch.tensor(input_token).type(torch.LongTensor)
			
			if(word==self.config.end_word):
				break
			else:
				words.append(word)

		return ' '.join(words)