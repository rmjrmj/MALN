# coding=utf-8
"""PyTorch BERT model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
import time

from MALN_model.until_module import LayerNorm
from MALN_model.until_module import GradReverse

import warnings
warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

import argparse
import csv
import logging
import os
import random
import pickle
import sys
# from global_config import *
import math
from torch.nn.utils.rnn import pad_sequence
import copy

import numpy as np 
from sklearn.metrics.pairwise import cosine_distances
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    AlbertModel,
    AlbertPreTrainedModel,
    AlbertConfig,
    load_tf_weights_in_albert,
)
from einops import rearrange, reduce, repeat


class MatchingAttention(nn.Module):
    
    def __init__(self, mem_dim, cand_dim, alpha_dim=None, att_type='general'):
        super(MatchingAttention, self).__init__()
        assert att_type!='concat' or alpha_dim!=None
        assert att_type!='dot' or mem_dim==cand_dim
        self.mem_dim = mem_dim
        self.cand_dim = cand_dim
        self.att_type = att_type
        if att_type=='general':
            self.transform = nn.Linear(cand_dim, mem_dim, bias=False)
        if att_type=='general2':
            self.transform = nn.Linear(cand_dim, mem_dim, bias=True)
            #torch.nn.init.normal_(self.transform.weight,std=0.01)
        elif att_type=='concat':
            self.transform = nn.Linear(cand_dim+mem_dim, alpha_dim, bias=False)
            self.vector_prod = nn.Linear(alpha_dim, 1, bias=False)

    def forward(self, M, x, mask=None):
        """
        M -> (seq_len, batch, mem_dim)
        x -> (batch, cand_dim)
        mask -> (batch, seq_len)
        """
        if type(mask)==type(None):
            mask = torch.ones(M.size(1), M.size(0)).type(M.type())

        if self.att_type=='dot':
            # vector = cand_dim = mem_dim
            M_ = M.permute(1,2,0) # batch, vector, seqlen
            x_ = x.unsqueeze(1) # batch, 1, vector
            alpha = F.softmax(torch.bmm(x_, M_), dim=2) # batch, 1, seqlen
        elif self.att_type=='general':
            M_ = M.permute(1,2,0) # batch, mem_dim, seqlen
            x_ = self.transform(x).unsqueeze(1) # batch, 1, mem_dim
            alpha = F.softmax(torch.bmm(x_, M_), dim=2) # batch, 1, seqlen
        elif self.att_type=='general2':
            M_ = M.permute(1,2,0) # batch, mem_dim, seqlen
            x_ = self.transform(x).unsqueeze(1) # batch, 1, mem_dim
            mask_ = mask.unsqueeze(2).repeat(1, 1, self.mem_dim).transpose(1, 2) # batch, seq_len, mem_dim
            M_ = M_ * mask_
            alpha_ = torch.bmm(x_, M_)*mask.unsqueeze(1)
            alpha_ = torch.tanh(alpha_)
            alpha_ = F.softmax(alpha_, dim=2)
            # alpha_ = F.softmax((torch.bmm(x_, M_))*mask.unsqueeze(1), dim=2) # batch, 1, seqlen
            alpha_masked = alpha_*mask.unsqueeze(1) # batch, 1, seqlen
            alpha_sum = torch.sum(alpha_masked, dim=2, keepdim=True) # batch, 1, 1
            alpha = alpha_masked/alpha_sum # batch, 1, 1 ; normalized
            #import ipdb;ipdb.set_trace()
        else:
            M_ = M.transpose(0,1) # batch, seqlen, mem_dim
            x_ = x.unsqueeze(1).expand(-1,M.size()[0],-1) # batch, seqlen, cand_dim
            M_x_ = torch.cat([M_,x_],2) # batch, seqlen, mem_dim+cand_dim
            mx_a = F.tanh(self.transform(M_x_)) # batch, seqlen, alpha_dim
            alpha = F.softmax(self.vector_prod(mx_a),1).transpose(1,2) # batch, 1, seqlen

        attn_pool = torch.bmm(alpha, M.transpose(0,1))[:,0,:] # batch, mem_dim
        return attn_pool, alpha

class GRUModel(nn.Module):
    
    def __init__(self, D_m, D_e, D_h, n_classes=6, dropout=0.5):
        
        super(GRUModel, self).__init__()
        
        self.n_classes = n_classes
        self.dropout   = nn.Dropout(dropout)
        self.gru = nn.GRU(input_size=D_m, hidden_size=D_e, num_layers=2, bidirectional=True, dropout=dropout)
        self.matchatt = MatchingAttention(2*D_e, 2*D_e, att_type='general2')
        self.linear = nn.Linear(2*D_e, D_h)
        self.smax_fc = nn.Linear(D_h, n_classes)
        
    def forward(self, U, qmask, umask, att2=False):
        """
        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party
        """
        emotions, hidden = self.gru(U)
        alpha, alpha_f, alpha_b = [], [], []
        
        if att2:
            att_emotions = []
            alpha = []
            for t in emotions:
                att_em, alpha_ = self.matchatt(emotions,t,mask=umask)
                att_emotions.append(att_em.unsqueeze(0))
                alpha.append(alpha_[:,0,:])
            att_emotions = torch.cat(att_emotions,dim=0)
            hidden = F.relu(self.linear(att_emotions))
        else:
            hidden = F.relu(self.linear(emotions))
        
        # hidden = F.relu(self.linear(emotions))
        hidden = self.dropout(hidden)
        # log_prob = F.log_softmax(self.smax_fc(hidden), 2)
        # return log_prob, alpha, alpha_f, alpha_b, emotions
        return hidden


class LSTMModel(nn.Module):
    
    def __init__(self, D_m, D_e, D_h, n_classes=6, dropout=0.5):
        
        super(LSTMModel, self).__init__()
        
        self.n_classes = n_classes
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_size=D_m, hidden_size=D_e, num_layers=2, bidirectional=True, dropout=dropout)
        self.matchatt = MatchingAttention(2*D_e, 2*D_e, att_type='general2')
        self.linear = nn.Linear(2*D_e, D_h)
        self.smax_fc = nn.Linear(D_h, n_classes)

    def forward(self, U, qmask, umask, att2=False):
        """
        U -> seq_len, batch, D_m
        qmask -> seq_len, batch, party
        """
        emotions, hidden = self.lstm(U)
        alpha, alpha_f, alpha_b = [], [], []
        
        if att2:
            att_emotions = []
            alpha = []
            for t in emotions:
                att_em, alpha_ = self.matchatt(emotions, t, mask=umask)
                att_emotions.append(att_em.unsqueeze(0))
                alpha.append(alpha_[:,0,:])
            att_emotions = torch.cat(att_emotions,dim=0)
            hidden = F.relu(self.linear(att_emotions))
        else:
            hidden = F.relu(self.linear(emotions))
        
        # hidden = F.relu(self.linear(emotions))
        hidden = self.dropout(hidden)
        #log_prob = F.log_softmax(self.smax_fc(hidden), 2)
        return hidden


class MaskedNLLLoss(nn.Module):
    
    def __init__(self, weight=None):
        super(MaskedNLLLoss, self).__init__()
        self.weight = weight
        self.loss = nn.NLLLoss(weight=weight,
                               reduction='sum')

    def forward(self, pred, target, mask):
        """
        pred -> batch*seq_len, n_classes
        target -> batch*seq_len
        mask -> batch, seq_len
        """
        mask_ = mask.view(-1,1) # batch*seq_len, 1
        if type(self.weight)==type(None):
            loss = self.loss(pred*mask_, target)/torch.sum(mask)
        else:
            loss = self.loss(pred*mask_, target)\
                            /torch.sum(self.weight[target]*mask_.squeeze())
        return loss


class Transformer(nn.Module):
    def __init__(self, d_model, num_layers=1, nhead=1, dropout=0.1, dim_feedforward=128, max_seq_length=5000):
        super(Transformer, self).__init__()
        self.d_model = d_model
        self.pos_encoder = nn.Embedding(max_seq_length, d_model)
        self.encoder = TransformerEncoder(TransformerLayer(d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout), num_layers=num_layers)
        self.decoder = nn.Linear(d_model, 1)
        self.norm = nn.LayerNorm(d_model)
        self.cls_token = nn.Parameter(torch.randn(1,1, d_model))

    def forward(self, input, d_model, attention_mask=None):
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=input.size()[0])
        seq_length = input.size()[1]
        position_ids = torch.arange(seq_length+1, dtype=torch.long, device=input.device)
        # positions_embedding = self.pos_encoder(position_ids).unsqueeze(0).expand(input.size()) # (seq_length, d_model) => (batch_size, seq_length, d_model)
        positions_embedding = self.pos_encoder(position_ids).unsqueeze(0).expand([input.size()[0], int(seq_length)+1, d_model]) # (seq_length, d_model) => (batch_size, seq_length, d_model)
        input = torch.cat([cls_tokens, input], dim=1)
        input = input + positions_embedding
        input = self.norm(input)
        hidden = self.encoder(input, attention_mask=attention_mask)
        # out = self.decoder(hidden) # (batch_size, seq_len, hidden_dim)
        # out = (out[:,0,:], out, hidden) # ([CLS] token embedding, full output, last hidden layer)
        # out = (hidden[:,0,:], hidden[:,1:,:])
        out = hidden[:,0,:]
        return out


class TransformerLayer(nn.Module):
    def __init__(self, hidden_size, nhead=1, dim_feedforward=128, dropout=0.1):
        super(TransformerLayer, self).__init__()
        self.self_attention = Attention(hidden_size, nhead, dropout)
        self.fc = nn.Sequential(nn.Linear(hidden_size, dim_feedforward), nn.ReLU(), nn.Linear(dim_feedforward, hidden_size))
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, attention_mask=None):
        src_1 = self.self_attention(src, src, attention_mask=attention_mask)
        src = src + self.dropout1(src_1)
        src = self.norm1(src)
        src_2 = self.fc(src)
        src = src + self.dropout2(src_2)
        src = self.norm2(src)

        return src


class TransformerEncoder(nn.Module):
    def __init__(self, layer, num_layers):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(layer, num_layers)
    def forward(self, src, attention_mask=None):
        for layer in self.layers:
            new_src = layer(src, attention_mask=attention_mask)
            src = src + new_src
        return src


class Attention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob, ctx_dim=None):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # visual_dim = 2048
        if ctx_dim is None:
            ctx_dim = hidden_size
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(ctx_dim, self.all_head_size)
        self.value = nn.Linear(ctx_dim, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, context, attention_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(context)
        mixed_value_layer = self.value(context)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Apply the attention mask is 
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class CrossAttentionLayer(nn.Module):
    def __init__(self, hidden_size, context_size, nhead=1, dropout=0.1):
        super(CrossAttentionLayer, self).__init__()
        self.src_cross_attention = Attention(hidden_size, nhead, dropout, ctx_dim=context_size)
        self.context_cross_attention = Attention(context_size, nhead, dropout, ctx_dim=hidden_size)
        self.self_attention = Attention(hidden_size + context_size, nhead, dropout)
        self.fc = nn.Sequential(nn.Linear(hidden_size + context_size, hidden_size + context_size), nn.ReLU())
        self.norm1 = nn.LayerNorm(hidden_size + context_size)
        self.norm2 = nn.LayerNorm(hidden_size + context_size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, context, attention_mask=None):
        new_src = self.src_cross_attention(src, context, attention_mask=attention_mask)
        new_context = self.context_cross_attention(context, src, attention_mask=attention_mask)
        
        cross_src = torch.cat((new_src, new_context), dim=2)
        # cross_src = torch.cat((cross_src, common_feature), dim=2)
        
        
        cross_src_1 = self.self_attention(cross_src, cross_src, attention_mask)
        cross_src = cross_src + self.dropout1(cross_src_1)
        cross_src = self.norm1(cross_src)

        cross_src_2 = self.fc(cross_src)
        cross_src = cross_src + self.dropout2(cross_src_2)
        cross_src = self.norm2(cross_src)

        return cross_src

class CrossAttentionEncoder(nn.Module):
    def __init__(self, layer, num_layers):
        super(CrossAttentionEncoder, self).__init__()
        self.layers = _get_clones(layer, num_layers)

    def forward(self, src, context, attention_mask=None):
        src_dim = src.size()[2]
        context_dim = context.size()[2]

        for layer in self.layers:
            output = layer(src, context, attention_mask=attention_mask)
            new_src = output[:,:,0:src_dim]
            new_context = output[:,:,src_dim:src_dim+context_dim]

            src = src + new_src
            context = context + new_context

        return output


class SpeakerEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self):
        super(SpeakerEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(2, 100)
        self.LayerNorm = LayerNorm(100, eps=1e-12)
        self.dropout = nn.Dropout(0.3)

    def forward(self, label_input): #[B, 7]
        seq_length = label_input.size(1)
        words_embeddings = self.word_embeddings(label_input)
        embeddings = words_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class hetero_loss(nn.Module):
    def __init__(self, margin=0.1, dist_type = 'l2'):
        super(hetero_loss, self).__init__()
        self.margin = margin
        self.dist_type = dist_type
        self.dist = nn.MSELoss(reduction='sum')
		# if dist_type == 'l2':
        #     self.dist = nn.MSELoss(reduction='sum')
		# if dist_type == 'cos':
        #     self.dist = nn.CosineSimilarity(dim=0)
		# if dist_type == 'l1':
        #     self.dist = nn.L1Loss()
        
    def forward(self, feat1, feat2, label1):

        label_num = len(label1.unique())
        
        feat1 = feat1.reshape(-1, feat1.shape[-1])
        feat2 = feat2.reshape(-1, feat2.shape[-1])
        
        feat1 = feat1.chunk(label_num, 0)
        feat2 = feat2.chunk(label_num, 0)
        
        
		#loss = Variable(.cuda())
        for i in range(label_num):
            center1 = torch.mean(feat1[i], dim=0)
            center2 = torch.mean(feat2[i], dim=0)
            if self.dist_type == 'l2' or self.dist_type == 'l1':
                if i == 0:
                    dist = max(0, abs(self.dist(center1, center2)))
                else:
                    dist += max(0, abs(self.dist(center1, center2)))
            elif self.dist_type == 'cos':
                if i == 0:
                    dist = max(0, 1-self.dist(center1, center2))
                else:
                    dist += max(0, 1-self.dist(center1, center2))
        
        return dist*0.1


class MALN(nn.Module):
    
    def __init__(self, args, dropout=0.1,fusion_dim=1024):
        
        super(MALN, self).__init__()
        self.newly_added_config=args
        LANGUAGE_DIM = 1024
        ACOUSTIC_DIM = 100
        VISUAL_DIM = 512

        # Define the speaker embedding layers
        self.SpeakerEmbedding = SpeakerEmbeddings()
        self.Text_Topic_Transformer_Speaker = Transformer(d_model=1024)
        self.Text_Topic_Transformer_Lisener = Transformer(d_model=1024)
        
        self.Acoustic_Topic_Transformer_Speaker = Transformer(d_model=100)
        self.Acoustic_Topic_Transformer_Lisener = Transformer(d_model=100)
        
        self.Visual_Topic_Transformer_Speaker = Transformer(d_model=512)
        self.Visual_Topic_Transformer_Lisener = Transformer(d_model=512)
        
        # Define the cross attention layer
        self.L_AV_layer = CrossAttentionLayer(100, 100, nhead=args.cross_n_heads, dropout=args.dropout)
        self.L_AV = CrossAttentionEncoder(self.L_AV_layer, args.cross_n_layers)
        self.selfattentionlayer = TransformerLayer(dim_feedforward=300,hidden_size=300)

        # Define the FC layers
        self.TextTopicFc = nn.Linear(1024, 100)
        self.AcousticTopicFc = nn.Linear(100, 100)
        self.VisualTopicFc = nn.Linear(512, 100)
        
        self.fc_visual = nn.Sequential(nn.Linear(100, 100), 
                                nn.ReLU(), 
                                nn.Dropout(args.dropout),
                                nn.Linear(100, 6))
        
        self.fc_text = nn.Sequential(nn.Linear(100, 100), 
                                nn.ReLU(), 
                                nn.Dropout(args.dropout),
                                nn.Linear(100, 6))
        
        self.fc_audio = nn.Sequential(nn.Linear(100, 100), 
                                nn.ReLU(), 
                                nn.Dropout(args.dropout),
                                nn.Linear(100, 6))
        
        self.fc = nn.Sequential(nn.Linear(300, 100), 
                                nn.ReLU(), 
                                nn.Dropout(args.dropout),
                                nn.Linear(100, 6))
        
        self.Acoustic_FC = nn.Linear(100, 512)
        self.Fusion_FC = nn.Linear(1024+100+512, 1024)
        self.Text_FC = nn.Linear(1024+768, 100+512)
        self.AVOut_FC = nn.Linear(100+512, 512)

        self.AVTOut_FC = nn.Linear(300, 100)

        # Define the GRU models
        self.TexGRU = GRUModel(D_m = 1024, D_h = 1024, D_e = 1024)
        self.VisualGRU = GRUModel(D_m = 512, D_h = 512, D_e = 512)
        self.AcousticGRU = GRUModel(D_m = 100, D_h = 100, D_e = 100)
        
        # Define the adversarial layers
        self.private_feature_extractor_text = nn.Sequential(
            nn.Linear(1024+100, 100),
            nn.Dropout(p=0.1),
            nn.Tanh())        
        self.private_feature_extractor_visual = nn.Sequential(
            nn.Linear(512+100, 100),
            nn.Dropout(p=0.1),
            nn.Tanh())
        self.private_feature_extractor_acoustic = nn.Sequential(
            nn.Linear(100+100, 100),
            nn.Dropout(p=0.1),
            nn.Tanh())
        
        self.common_feature_extractor_text= nn.Sequential(
            nn.Linear(1024+100, 100),
            nn.Dropout(p=0.3),
            nn.Tanh()
        )

        
        self.common_feature_extractor_visual= nn.Sequential(
            nn.Linear(512+100, 100),
            nn.Dropout(p=0.3),
            nn.Tanh()
        )
        
        self.common_feature_extractor_acoustic= nn.Sequential(
            nn.Linear(100+100, 100),
            nn.Dropout(p=0.3),
            nn.Tanh()
        )

        self.common_feature_extractor_share= nn.Sequential(
            nn.Linear(100, 100),
            nn.Dropout(p=0.3),
            nn.Tanh()
        )
        

        self.modal_discriminator = nn.Sequential(
            nn.Linear(100, 100 // 2),
            nn.Dropout(p=0.1),
            nn.ReLU(),
            nn.Linear(100 // 2, 3),
        )
        
        # self.cross_classifier = EmotionClassifier(cross_config.hidden_size, 1) 
        self.ml_loss = nn.BCELoss()
        self.adv_loss = nn.CrossEntropyLoss()
        
        self.av_fc = nn.Linear(200, 100)
        self.av_fc2 = nn.Linear(200, 100)
        
        self.common_classfier = nn.Sequential(
            nn.Linear(100, 6),
            nn.Dropout(0.1),
            nn.Sigmoid()
        )
        
    def get_params(self):
        
        acoustic_params=list(self.acoustic_model.named_parameters())
        visual_params=list(self.visual_model.named_parameters())
        
        other_params=list(self.text_model.named_parameters())+list(self.L_AV.named_parameters())+list(self.fc.named_parameters())
        
        return acoustic_params,visual_params,other_params
    
    def calculate_orthogonality_loss(self, first_feature, second_feature):
        diff_loss = torch.norm(torch.bmm(first_feature, second_feature.transpose(1, 2)), dim=(1, 2)).pow(2).mean()
        return diff_loss
    
    def _get_cross_output(self, sequence_output, visual_output, audio_output, common_feature, attention_mask, visual_mask, audio_mask, common_mask):
    
        # =============> visual audio fusion
        va_concat_features = torch.cat((audio_output, visual_output), dim=1)
        va_concat_mask = torch.cat((audio_mask, visual_mask), dim=1)
        text_type_ = torch.zeros_like(attention_mask)
        video_type_ = torch.ones_like(visual_mask)
        audio_type_ =  torch.zeros_like(audio_mask)
        va_concat_type = torch.cat((audio_type_, video_type_), dim=1)
        va_cross_layers, va_pooled_output = self.va_cross(va_concat_features, va_concat_type, va_concat_mask)
        va_cross_output = va_cross_layers[-1]
        # <============= visual audio fusion

        # =============> VisualAudio and text fusion
        vat_concat_features = torch.cat((sequence_output, va_cross_output), dim=1)
        vat_concat_mask = torch.cat((attention_mask, va_concat_mask), dim=1)
        va_type_ = torch.ones_like(va_concat_mask)
        vat_concat_type = torch.cat((text_type_, va_type_), dim=1)
        vat_cross_layers, vat_pooled_output = self.vat_cross(vat_concat_features, vat_concat_type, vat_concat_mask)
        vat_cross_output = vat_cross_layers[-1]
        # <============= VisualAudio and text fusion

        # =============> private common fusion
        pc_concate_features = torch.cat((vat_cross_output, common_feature), dim=1)
        specific_type = torch.zeros_like(vat_concat_mask)
        common_type = torch.ones_like(common_mask)
        pc_concate_type = torch.cat((specific_type, common_type), dim=1)
        pc_concat_mask = torch.cat((vat_concat_mask, common_mask), dim=1)
        pc_cross_layers, pc_pooled_output = self.pc_cross(pc_concate_features, pc_concate_type, pc_concat_mask)
        pc_cross_output = pc_cross_layers[-1]
        # <============= private common fusion
 
        return  pc_pooled_output, pc_cross_output, pc_concat_mask
    

    def forward(self, training, groundTruth_labels, text, visual, acoustic, x1, x2, x3, x4, x5, x6, o1, o2, o3, qmask, umask, attention_mask=None, token_type_ids=None):
        text_topic_data = text
        visual_topic_data = visual
        acoustic_topic_data = acoustic    
        
        text_topic_data_speaker = text_topic_data_lisener = text_topic_data.permute(1, 0, 2)
        visual_topic_data_speaker = visual_topic_data_lisener = visual_topic_data.permute(1, 0, 2)
        acoustic_topic_data_speaker = acoustic_topic_data_lisener = acoustic_topic_data.permute(1, 0, 2)
        
        SpeakerMask = qmask.permute(1,0,2)
        SpeakerMask = torch.max(SpeakerMask, dim=-1)[1]
        SpeakerMask = SpeakerMask.unsqueeze(-1)
        
        Lisener_Mask = qmask.permute(1,0,2)
        Lisener_Mask = torch.min(Lisener_Mask, dim=-1)[1]
        Lisener_Mask = Lisener_Mask.unsqueeze(-1)

        # speaker information embedding process
        text_topic_data_speaker = self.Text_Topic_Transformer_Speaker(text_topic_data_speaker, d_model=1024)
        text_topic_data_lisener = self.Text_Topic_Transformer_Lisener(text_topic_data_lisener, d_model=1024)
        
        text_topic_data_speaker_repeat = text_topic_data_speaker.repeat(text.shape[0], 1, 1)
        text_topic_data_lisener_repeat = text_topic_data_lisener.repeat(text.shape[0], 1, 1)
        
        text_topic = text_topic_data_speaker_repeat *  SpeakerMask.permute(1,0,2) + text_topic_data_lisener_repeat * Lisener_Mask.permute(1,0,2)
        text_topic = self.TextTopicFc(text_topic)
        
        acoustic_topic_data_speaker = self.Acoustic_Topic_Transformer_Speaker(acoustic_topic_data_speaker, d_model=100)
        acoustic_topic_data_lisener = self.Acoustic_Topic_Transformer_Lisener(acoustic_topic_data_lisener, d_model=100)
        
        acoustic_topic_data_speaker_repeat = acoustic_topic_data_speaker.repeat(acoustic.shape[0], 1, 1)
        acoustic_topic_data_lisener_repeat = acoustic_topic_data_lisener.repeat(acoustic.shape[0], 1, 1)
        
        acoustic_topic = acoustic_topic_data_speaker_repeat * SpeakerMask.permute(1,0,2) + acoustic_topic_data_lisener_repeat * Lisener_Mask.permute(1,0,2)
        acoustic_topic = self.AcousticTopicFc(acoustic_topic)
        
        visual_topic_data_speaker = self.Visual_Topic_Transformer_Speaker(visual_topic_data_speaker, d_model=512)
        visual_topic_data_lisener = self.Visual_Topic_Transformer_Lisener(visual_topic_data_lisener, d_model=512)
        
        visual_topic_data_speaker_repeat = visual_topic_data_speaker.repeat(visual.shape[0], 1, 1)
        visual_topic_data_lisener_repeat = visual_topic_data_lisener.repeat(visual.shape[0], 1, 1)
        
        visual_topic = visual_topic_data_speaker_repeat * SpeakerMask.permute(1,0,2) + visual_topic_data_lisener_repeat * Lisener_Mask.permute(1,0,2)
        visual_topic = self.VisualTopicFc(visual_topic)

        # text_topic, visual_topic, acoustic_topic are the generated speaker information in each modality

        text = self.TexGRU(text, qmask, umask)
        visual = self.VisualGRU(visual, qmask, umask)
        acoustic = self.AcousticGRU(acoustic, qmask, umask)
        
        # Concatenate the speaker information and the content
        text = torch.cat((text, text_topic),dim=-1)
        visual = torch.cat((visual, visual_topic),dim=-1)
        acoustic = torch.cat((acoustic, acoustic_topic),dim=-1)
        
        text = text.permute(1,0,2)
        visual = visual.permute(1,0,2)
        acoustic = acoustic.permute(1,0,2)

        # private and common encoders
        
        private_text = self.private_feature_extractor_text(text)
        private_visual = self.private_feature_extractor_visual(visual)
        private_audio = self.private_feature_extractor_acoustic(acoustic)

        common_text = self.common_feature_extractor_text(text)
        common_visual = self.common_feature_extractor_visual(visual)
        common_audio = self.common_feature_extractor_acoustic(acoustic)

        common_text = self.common_feature_extractor_share(common_text)
        common_visual = self.common_feature_extractor_share(common_visual)
        common_audio = self.common_feature_extractor_share(common_audio)

        # common feature
        common_feature = common_text + common_visual + common_audio

        visual_output = common_visual
        text_output = common_text
        audio_output = common_audio

        out_visual = self.fc_visual(visual_output)
        out_visual = F.log_softmax(out_visual, 2)
        out_visual = out_visual.permute(1,0,2)
        
        out_text = self.fc_text(text_output)
        out_text = F.log_softmax(out_text, 2)
        out_text = out_text.permute(1,0,2)
        
        out_audio = self.fc_audio(audio_output)
        out_audio = F.log_softmax(out_audio, 2)
        out_audio = out_audio.permute(1,0,2)

        DistanceCenter = 0

        # feature fusion process
    
        av_output=torch.cat((audio_output, visual_output),dim=2)
        av_output = self.av_fc(av_output)
        noverbal_text = self.L_AV(text_output, av_output, attention_mask=None)
        
        noverbal_text = torch.cat((noverbal_text, common_feature),dim=2)
        L_AV_embedding = noverbal_text
        # L_AV_embedding = self.selfattentionlayer(noverbal_text)
        out = self.fc(L_AV_embedding)
        out = F.log_softmax(out, 2)
        out = out.permute(1,0,2)
        
        common_mask = torch.ones_like(umask)
        
        # adversarial learning
        if training:
            text_modal = torch.zeros_like(common_mask).view(-1) #[B, L]
            visual_modal = torch.ones_like(common_mask).view(-1) #[B, L]
            audio_modal = visual_modal.data.new(visual_modal.size()).fill_(2) #[B, L]
            
            text_modal = torch.as_tensor(text_modal).long()
            visual_modal = torch.as_tensor(visual_modal).long()
            audio_modal = torch.as_tensor(audio_modal).long()

            private_text_modal_pred = self.modal_discriminator(private_text).view(-1, 3)
            private_visual_modal_pred = self.modal_discriminator(private_visual).view(-1, 3)
            private_audio_modal_pred = self.modal_discriminator(private_audio).view(-1, 3)

            # modality discriminator
            common_text_modal_pred = self.modal_discriminator(GradReverse.grad_reverse(common_text, 1)).view(-1, 3)
            common_visual_modal_pred = self.modal_discriminator(GradReverse.grad_reverse(common_visual, 1)).view(-1, 3)
            common_audio_modal_pred = self.modal_discriminator(GradReverse.grad_reverse(common_audio, 1)).view(-1, 3)

            all_loss = 0.
            # common classifier
            common_pred = self.common_classfier(common_feature)
            
            common_pred = F.log_softmax(common_pred, 2)
            common_pred = common_pred.permute(1,0,2)

            # calculate the orthogonality loss, including the private and common features
            
            preivate_diff_loss = self.calculate_orthogonality_loss(private_text, private_visual) + self.calculate_orthogonality_loss(private_text, private_audio) + self.calculate_orthogonality_loss(private_visual, private_audio)
            common_diff_loss = self.calculate_orthogonality_loss(common_text, private_text) + self.calculate_orthogonality_loss(common_visual, private_visual) + self.calculate_orthogonality_loss(common_audio, private_audio)
            
            # The type and ratio of the loss function can be adjusted according to the actual application scenario

            adv_preivate_loss = self.adv_loss(private_text_modal_pred, text_modal) + self.adv_loss(private_visual_modal_pred, visual_modal) + self.adv_loss(private_audio_modal_pred, audio_modal)
            adv_common_loss = self.adv_loss(common_text_modal_pred, text_modal) + self.adv_loss(common_visual_modal_pred, visual_modal) + self.adv_loss(common_audio_modal_pred, audio_modal)

            all_loss = 5e-6 * (preivate_diff_loss + common_diff_loss) + 0.01 * (adv_common_loss + adv_preivate_loss)

            return  all_loss, L_AV_embedding, out, out_audio, out_text, out_visual, common_pred, DistanceCenter, private_text, private_visual, private_audio, common_feature, common_text, common_visual, common_audio
        else:
            return L_AV_embedding, out, out_audio, out_text, out_visual, DistanceCenter, private_text, private_visual, private_audio, common_feature, common_text, common_visual, common_audio
            # return predict_labels, groundTruth_labels, predict_scores
    
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])