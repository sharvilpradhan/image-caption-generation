import torch

class Config:
	def __init__(self):
		self.use_gpu = torch.cuda.is_available()
		self.end_word = '<END>'
		self.start_word = '<START>'
		self.pad_word = '<PAD>'
		self.height = 299
		self.width = 299
		self.input_embedding = 300
		self.hidden_size = 300
		self.output_embedding = 300

		self.captions_file = '../caption_datasets/dataset_flickr8k.json'
		self.image_dir = '../Flicker8k_Dataset/'

		self.dropout = 0.22
		self.gru_layers = 3

		self.trained_model = '../trained_models/nnmodel.pth'
		self.inception_model = '../trained_models/inception.pth'
		self.use_IC_V6 = True