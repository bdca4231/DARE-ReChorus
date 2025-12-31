import torch
import torch.nn as nn
import torch.nn.functional as fn
import numpy as np
import math

from models.BaseContextModel import ContextSeqModel, ContextSeqCTRModel
from utils.layers import MLP_Block

class DAREBase(object):
	@staticmethod
	def parse_model_args_dare(parser):
		parser.add_argument('--emb_size', type=int, default=64,
							help='Size of embedding vectors for representation.')
		parser.add_argument('--att_emb_size', type=int, default=64,
							help='Size of embedding vectors for attention. If different from emb_size, enables dimension reduction for attention.')
		parser.add_argument('--att_layers', type=str, default='[64]',
							help="Size of each layer in the attention module (not used in DARE, kept for compatibility).")
		parser.add_argument('--dnn_layers', type=str, default='[64]',
							help="Size of each layer in the MLP module.")
		parser.add_argument('--use_time_mode', type=str, default='concat',
							help="How to incorporate time embedding: 'concat' or 'add'.")
		parser.add_argument('--time_emb_size', type=int, default=16,
							help='Size of time embedding vectors.')
		return parser

	def _define_init(self, args, corpus):
		self.user_context = ['user_id']+corpus.user_feature_names
		self.item_context = ['item_id']+corpus.item_feature_names
		self.situation_context = corpus.situation_feature_names
		self.item_feature_num = len(self.item_context)
		self.user_feature_num = len(self.user_context)
		self.situation_feature_num = len(corpus.situation_feature_names) if self.add_historical_situations else 0
  
		self.vec_size = args.emb_size
		self.att_vec_size = args.att_emb_size
		self.att_layers = eval(args.att_layers)
		self.dnn_layers = eval(args.dnn_layers)
		self.use_time_mode = args.use_time_mode
		self.time_emb_size = args.time_emb_size
		self._define_params_DARE()
		self.apply(self.init_weights)

	def _define_params_DARE(self):
		# Two separate embedding dictionaries: one for attention, one for representation
		self.embedding_dict_att = nn.ModuleDict()
		self.embedding_dict_rep = nn.ModuleDict()
		
		for f in self.user_context+self.item_context+self.situation_context:
			# Attention embeddings
			self.embedding_dict_att[f] = nn.Embedding(self.feature_max[f], self.att_vec_size) if f.endswith('_c') or f.endswith('_id') else\
					nn.Linear(1, self.att_vec_size, bias=False)
			# Representation embeddings
			self.embedding_dict_rep[f] = nn.Embedding(self.feature_max[f], self.vec_size) if f.endswith('_c') or f.endswith('_id') else\
					nn.Linear(1, self.vec_size, bias=False)

		# DARE uses scaled dot product attention, no MLP for attention
		# We set att_mlp_layers to None to indicate it's not used
		self.att_mlp_layers = None

		pre_size = (2*(self.item_feature_num+self.situation_feature_num)+self.item_feature_num
              +len(self.situation_context) + self.user_feature_num) * self.vec_size
		self.dnn_mlp_layers = MLP_Block(input_dim=pre_size, hidden_units=self.dnn_layers, output_dim=1,
                                  hidden_activations='Dice', dropout_rates=self.dropout, batch_norm=True,
                                  norm_before_activation=True)

	def attention(self, queries, keys, values, keys_length, mask_mat, softmax_stag=False, return_seq_weight=False):
		'''Scaled dot product attention as in DARE.
		queries: batch * (if*vecsize) for attention embeddings
		keys: batch * seq_len * (if*vecsize) for attention embeddings
		values: batch * seq_len * (if*vecsize) for representation embeddings (used as value)
		'''
		embedding_size = queries.shape[-1]  # H
		hist_len = keys.shape[1]  # T
		queries = queries.unsqueeze(1)  # batch * 1 * H
		
		# Scaled dot product
		scores = torch.matmul(queries, keys.transpose(1, 2)) / math.sqrt(float(embedding_size))
		scores = scores.squeeze(1)  # batch * seq_len
		
		# get mask
		mask = mask_mat.repeat(scores.size(0), 1)
		mask = mask >= keys_length.unsqueeze(1)
		# mask
		if softmax_stag:
			mask_value = -np.inf
		else:
			mask_value = 0.0
		scores = scores.masked_fill(mask=mask, value=torch.tensor(mask_value))
		scores = scores.unsqueeze(1)  # batch * 1 * seq_len
		
		# get the weight of each user's history list about the target item
		if softmax_stag:
			weights = fn.softmax(scores, dim=2)  # [B, 1, T]
		else:
			weights = scores
		
		if return_seq_weight:
			output = weights.squeeze(1)  # [B, T]
		else:
			# Use representation embeddings as value
			output = torch.matmul(weights, values)  # [B, 1, H]
			output = output.squeeze(dim=1)
		
		torch.cuda.empty_cache()
		return output

	def get_all_embedding(self, feed_dict, merge_all=True):
		# Attention embeddings
		item_feats_emb_att = torch.stack([self.embedding_dict_att[f](feed_dict[f]) if f.endswith('_c') or f.endswith('_id')
					else self.embedding_dict_att[f](feed_dict[f].float().unsqueeze(-1))
					for f in self.item_context], dim=-2) # batch * feature num * att_emb size
		history_item_emb_att = torch.stack([self.embedding_dict_att[f](feed_dict['history_'+f]) if f.endswith('_c') or f.endswith('_id')
					else self.embedding_dict_att[f](feed_dict['history_'+f].float().unsqueeze(-1))
					for f in self.item_context], dim=-2) # batch * feature num * att_emb size
		
		# Representation embeddings
		item_feats_emb_rep = torch.stack([self.embedding_dict_rep[f](feed_dict[f]) if f.endswith('_c') or f.endswith('_id')
					else self.embedding_dict_rep[f](feed_dict[f].float().unsqueeze(-1))
					for f in self.item_context], dim=-2) # batch * feature num * rep_emb size
		history_item_emb_rep = torch.stack([self.embedding_dict_rep[f](feed_dict['history_'+f]) if f.endswith('_c') or f.endswith('_id')
					else self.embedding_dict_rep[f](feed_dict['history_'+f].float().unsqueeze(-1))
					for f in self.item_context], dim=-2) # batch * feature num * rep_emb size
		
		# User embeddings (representation only, as in DIN)
		user_feats_emb = torch.stack([self.embedding_dict_rep[f](feed_dict[f]) if f.endswith('_c') or f.endswith('_id')
					else self.embedding_dict_rep[f](feed_dict[f].float().unsqueeze(-1))
					for f in self.user_context], dim=-2) # batch * feature num * emb size
		
		# Situation embeddings
		if len(self.situation_context):
			situ_feats_emb = torch.stack([self.embedding_dict_rep[f](feed_dict[f]) if f.endswith('_c') or f.endswith('_id')
					else self.embedding_dict_rep[f](feed_dict[f].float().unsqueeze(-1))
					for f in self.situation_context], dim=-2) # batch * feature num * emb size
		else:
			situ_feats_emb = None
		
		# Historical situation embeddings
		if self.add_historical_situations and len(self.situation_context):
			history_situ_emb = torch.stack([self.embedding_dict_rep[f](feed_dict['history_'+f]) if f.endswith('_c') or f.endswith('_id')
					else self.embedding_dict_rep[f](feed_dict['history_'+f].float().unsqueeze(-1))
					for f in self.situation_context], dim=-2) # batch * feature num * emb size
			history_emb_att = torch.cat([history_item_emb_att, history_situ_emb], dim=-2).flatten(start_dim=-2)
			history_emb_rep = torch.cat([history_item_emb_rep, history_situ_emb], dim=-2).flatten(start_dim=-2)
			item_num = item_feats_emb_rep.shape[1]
			current_emb_att = torch.cat([item_feats_emb_att, situ_feats_emb.unsqueeze(1).repeat(1,item_num,1,1)], dim=-2).flatten(start_dim=-2)
			current_emb_rep = torch.cat([item_feats_emb_rep, situ_feats_emb.unsqueeze(1).repeat(1,item_num,1,1)], dim=-2).flatten(start_dim=-2)
		else:
			history_emb_att = history_item_emb_att.flatten(start_dim=-2)
			history_emb_rep = history_item_emb_rep.flatten(start_dim=-2)
			current_emb_att = item_feats_emb_att.flatten(start_dim=-2)
			current_emb_rep = item_feats_emb_rep.flatten(start_dim=-2)

		if merge_all:
			item_num = item_feats_emb_rep.shape[1]
			if situ_feats_emb is not None:
				all_context = torch.cat([item_feats_emb_rep, user_feats_emb.unsqueeze(1).repeat(1,item_num,1,1),
							situ_feats_emb.unsqueeze(1).repeat(1,item_num,1,1)], dim=-2).flatten(start_dim=-2)
			else:
				all_context = torch.cat([item_feats_emb_rep, user_feats_emb.unsqueeze(1).repeat(1,item_num,1,1),
							], dim=-2).flatten(start_dim=-2)
					
			return history_emb_att, current_emb_att, history_emb_rep, current_emb_rep, all_context
		else:
			return history_emb_att, current_emb_att, history_emb_rep, current_emb_rep, user_feats_emb, situ_feats_emb

	def forward(self, feed_dict):
		hislens = feed_dict['lengths']
		history_emb_att, current_emb_att, history_emb_rep, current_emb_rep, all_context = self.get_all_embedding(feed_dict)
		predictions = self.att_dnn(current_emb_att, history_emb_att, current_emb_rep, history_emb_rep, all_context, hislens)
		return {'prediction': predictions}

	def att_dnn(self, current_emb_att, history_emb_att, current_emb_rep, history_emb_rep, all_context, history_lengths):
		mask_mat = (torch.arange(history_emb_att.shape[1]).view(1,-1)).to(self.device)
  
		batch_size, item_num, feats_emb_att = current_emb_att.shape
		_, max_len, his_emb_att = history_emb_att.shape
		current_emb_att_2d = current_emb_att.view(-1, feats_emb_att)
		history_emb_att_2d = history_emb_att.unsqueeze(1).repeat(1,item_num,1,1).view(-1,max_len,his_emb_att)
		hislens2d = history_lengths.unsqueeze(1).repeat(1,item_num).view(-1)
		
		# Representation embeddings for value
		_, _, feats_emb_rep = current_emb_rep.shape
		history_emb_rep_2d = history_emb_rep.unsqueeze(1).repeat(1,item_num,1,1).view(-1,max_len,feats_emb_rep)
		
		# Compute attention using attention embeddings (Q,K) and representation embeddings (V)
		# This returns weighted sum of representation embeddings
		weighted_rep = self.attention(current_emb_att_2d, history_emb_att_2d, history_emb_rep_2d, 
									  hislens2d, mask_mat, softmax_stag=True, return_seq_weight=False)
		
		current_emb_rep_2d = current_emb_rep.view(-1, feats_emb_rep)
		
		# Target-aware representation: element-wise product
		din_output = torch.cat([weighted_rep, weighted_rep*current_emb_rep_2d, all_context.view(batch_size*item_num,-1)], dim=-1)
		din_output = self.dnn_mlp_layers(din_output)
		return din_output.squeeze(dim=-1).view(batch_size, item_num)


class DARECTR(ContextSeqCTRModel, DAREBase):
	reader = 'ContextSeqReader'
	runner = 'CTRRunner'
	extra_log_args = ['emb_size', 'att_emb_size', 'att_layers', 'add_historical_situations', 'use_time_mode', 'time_emb_size']
	
	@staticmethod
	def parse_model_args(parser):
		parser = DAREBase.parse_model_args_dare(parser)
		return ContextSeqCTRModel.parse_model_args(parser)
	
	def __init__(self, args, corpus):
		ContextSeqCTRModel.__init__(self, args, corpus)
		self._define_init(args, corpus)

	def forward(self, feed_dict):
		out_dict = DAREBase.forward(self, feed_dict)
		out_dict['prediction'] = out_dict['prediction'].view(-1).sigmoid()
		out_dict['label'] = feed_dict['label'].view(-1)
		return out_dict

class DARETopK(ContextSeqModel, DAREBase):
	reader = 'ContextSeqReader'
	runner = 'BaseRunner'
	extra_log_args = ['emb_size', 'att_emb_size', 'att_layers', 'add_historical_situations', 'use_time_mode', 'time_emb_size']
	
	@staticmethod
	def parse_model_args(parser):
		parser = DAREBase.parse_model_args_dare(parser)
		return ContextSeqModel.parse_model_args(parser)
	
	def __init__(self, args, corpus):
		ContextSeqModel.__init__(self, args, corpus)
		self._define_init(args, corpus)

	def forward(self, feed_dict):
		return DAREBase.forward(self, feed_dict)
