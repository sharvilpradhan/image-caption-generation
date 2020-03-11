from config import Config
import torch.nn as nn
import torch

config = Config()

class NNModel(nn.Module):
        
    def __init__(self, token_dict_size):
        super(NNModel, self).__init__()        
        
        self.embedding_size = config.input_embedding
        self.hidden_state_size = config.hidden_size
        self.token_dict_size = token_dict_size
        self.output_size = config.output_embedding
        self.batchnorm = nn.BatchNorm1d(self.embedding_size)
        self.input_embedding = nn.Embedding(self.token_dict_size, self.embedding_size)
        self.embedding_dropout = nn.Dropout(p=0.22)
        self.gru_layers = 3
        self.gru = nn.GRU(input_size=self.embedding_size, hidden_size=self.hidden_state_size, num_layers=self.gru_layers, dropout=0.22)
        self.linear = nn.Linear(self.hidden_state_size, self.output_size)
        self.out = nn.Linear(self.output_size, token_dict_size)
        
    def forward(self, input_tokens, hidden, inception = None, process_image=False, use_inception=True):
        if(config.use_gpu):
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        
        if(process_image):
            if(use_inception):
                inp=self.embedding_dropout(inception(hidden))
            else:
                inp=hidden
            hidden=torch.zeros((self.gru_layers,1, self.hidden_state_size))
        else:
            inp=self.embedding_dropout(self.input_embedding(input_tokens.view(1).type(torch.LongTensor).to(device)))        
        hidden = hidden.view(self.gru_layers,1,-1)
        inp = inp.view(1,1,-1)
        out, hidden = self.gru(inp, hidden)
        out = self.out(self.linear(out))
        return out, hidden    