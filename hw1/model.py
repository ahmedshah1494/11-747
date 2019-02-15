import torch
from torch import nn
import torch.nn.functional as F
import copy

class EnsembleXEntropy(nn.Module):
    """docstring for EnsembleXEntropy"""
    def __init__(self, L=0.1):
        super(EnsembleXEntropy, self).__init__()
        self.L = L
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.avg_jsd = None        
    # assume outputs.shape = (B x C x K)
    # weights.shape = (K,)
    def forward(self, outputs, targets):
        targets = targets#.unsqueeze(1)  
        targets = targets#.repeat(1, outputs.shape[2])                
        outputs_ = torch.mean(outputs, 2)
        loss = self.criterion(outputs_, targets)
        # loss = self.nll(F.log_softmax(outputs, dim=1),targets)
        # loss = torch.mean(loss, 1)

        if self.L == 0:
            self.avg_jsd = 0
            return torch.mean(loss)

        p = F.softmax(outputs, dim=1)
        log_p = F.log_softmax(outputs, dim=1)
        ent_P = -1 * torch.sum(p * log_p, 1, keepdim=True) # Bx1xK
        if torch.isnan(ent_P).any():
            print ('p', p)
            print ('ent_p', ent_P)
            raise (ValueError)
        mean_ent_P = torch.mean(ent_P, 2, keepdim=True) # Bx1x1
        mean_P = torch.mean(p, 2, keepdim=True) # BxCx1
        ent_mean_P = -1 * torch.sum(mean_P * torch.log(mean_P), 1, keepdim=True) #Bx1x1
        if torch.isnan(ent_mean_P).any():
            print ('outputs', outputs)
            print ('ent_mean_p', ent_mean_P)
            raise (ValueError)
        jsd = ent_mean_P - mean_ent_P
        jsd = jsd.squeeze() 
        
        # print targets.shape
        
        # print jsd
        self.avg_jsd = torch.mean(jsd)

        loss -= self.L*jsd  
        return torch.mean(loss)

class Ensemble(nn.Module):
    """docstring for EnsembleMLP"""
    def __init__(self, models, n_classes):
        super(Ensemble, self).__init__()
        n_models = len(models)
        self.models = nn.ModuleList(models)

    def forward(self, x):
        # x = F.relu(self.drop(self.linear(x)))
        outputs = [model(x) for model in self.models]
        outputs = torch.stack(outputs, dim=2)              
        outputs_ = outputs 

        return outputs_

class CNN(nn.Module):
	"""docstring for CNN"""
	def __init__(self, nwords, emb_size, num_filters, window_sizes, ntags, init_embed=None):
		super(CNN, self).__init__()

		""" layers """		
		if init_embed is None:
			self.embedding = torch.nn.Embedding(nwords, emb_size)
			# uniform initialization
			torch.nn.init.uniform_(self.embedding.weight, -0.25, 0.25)
		else:
			self.embedding = torch.nn.Embedding(init_embed.shape[0], init_embed.shape[1])
			self.embedding.weight.data.copy_(init_embed)

		self.embedding_static = torch.nn.Embedding(nwords, emb_size)
		self.embedding_static.weight.data.copy_(self.embedding.weight.clone())
		self.embedding_static.weight.requires_grad = False
		# Conv 1d
		self.convs = nn.ModuleList([torch.nn.Conv1d(in_channels=emb_size, out_channels=num_filters, kernel_size=window_sizes[i],
		                           stride=1, padding=0, dilation=1, groups=1, bias=True) for i in range(len(window_sizes))])
		self.dropout = nn.Dropout(0.5)
		# self.fc = nn.Sequential(nn.Linear(num_filters*len(self.convs), 128),
		# 						nn.BatchNorm1d(128),
		# 						nn.ReLU(),
		# 						nn.Dropout(0.2),
		# 						nn.Linear(128,64),
		# 						nn.BatchNorm1d(64),
		# 						nn.ReLU(),
		# 						nn.Dropout(0.1),
		# 						nn.Linear(64,32),
		# 						nn.BatchNorm1d(32),
		# 						nn.ReLU(),)
		self.projection_layer = torch.nn.Linear(in_features=num_filters*len(self.convs), out_features=ntags, bias=True)	
		# Initializing the projection layer
		torch.nn.init.xavier_uniform_(self.projection_layer.weight)

		self.projection_layer_max_norm = 3

	def forward(self, words):		
		emb = self.embedding(words)
		emb_stat = self.embedding_static(words)

		emb = emb.permute(0, 2, 1)
		emb_stat = emb_stat.permute(0, 2, 1)    
		
		feats = [F.relu((c(emb)+c(emb_stat)).max(dim=2)[0]) for c in self.convs]
		feats = torch.cat(feats, dim=1)		
		feats = self.dropout(feats)
		# feats = self.fc(feats)

		if self.training and torch.norm(self.projection_layer.weight) > self.projection_layer_max_norm:
			with torch.no_grad():
				self.projection_layer.weight *= self.projection_layer_max_norm/torch.norm(self.projection_layer.weight) 
		out = self.projection_layer(feats)
		return out
		
		