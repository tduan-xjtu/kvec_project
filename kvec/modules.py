# -*- coding: utf-8 -*-

import math

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.modules.transformer import _get_activation_fn, _get_clones
from torch.nn import functional as F
from typing import Optional
from haltnet import Controller, BaselineNetwork

import pandas as pd
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ====================================================================================================
class FixedPositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=1024).
    """

    def __init__(self, d_model, dropout=0.1, max_len=256):
        super(FixedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # positional encoding
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)  # this stores the variable in the state_dict (used for non-trainable variables)

    def forward(self):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """
        return self.pe


class LearnablePositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super(LearnablePositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Each position gets its own embedding
        # Since indices are always 0 ... max_len, we don't have to do a look-up
        self.pe = nn.Parameter(torch.empty(max_len, 1, d_model))  # requires_grad automatically set to True
        nn.init.uniform_(self.pe, -0.02, 0.02)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def get_pos_encoder(pos_encoding):
    if pos_encoding == "learnable":
        return LearnablePositionalEncoding
    elif pos_encoding == "fixed":
        return FixedPositionalEncoding

    raise NotImplementedError("pos_encoding should be 'learnable'/'fixed', not '{}'".format(pos_encoding))
# ====================================================================================================
class My_Embedding(nn.Module):

    def __init__(self, pck_vocab_size=0, d_model=128, n_segments=2):
        super(My_Embedding, self).__init__()
        self.pck_vocab_size = pck_vocab_size
        self.d_model = d_model
        self.n_segments = n_segments

        self.pck_embed = nn.Embedding(pck_vocab_size, d_model)
        self.seg_embed = nn.Embedding(n_segments, d_model)
        # self.ret_pos_embed = FixedRelativePositionalEncoder(d_model=d_model,max_len=256)
        self.abs_pos_embed = FixedPositionalEncoding(d_model=d_model)

        self.norm = nn.LayerNorm(d_model)

    def FixedRelativePositionalEncoder(self, ret_pos, d_model=128, max_len=200):
        batch_size = ret_pos.shape[0]
        pe = torch.zeros(batch_size, max_len, d_model).to(device)

        relative_pos = ret_pos.unsqueeze(2)  # [batch_size,seq_len,1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)).to(device)
        # div_term.shape [d_model/2]

        pe[:, :, 0::2] = torch.sin(relative_pos * div_term)
        pe[:, :, 1::2] = torch.cos(relative_pos * div_term)

        return pe

    def forward(self, x, seg, ret_pos, value2token_offset):
        """

        :param x: value
        :param seg: key
        :param ret_pos: relative position
        :return: value_embedding + key_embedding + relative_position_embedding
        """
        embedding = self.pck_embed(x + value2token_offset) + self.seg_embed(seg) + \
                    self.FixedRelativePositionalEncoder(ret_pos) + self.abs_pos_embed()

        return self.norm(embedding)
# ====================================================================================================
class EncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(EncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None,
                ) -> [Tensor, Tensor]:  # output two tensor
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional). 传入kvb-mask shape: [batch_size*num_head,seq_len,seq_len]
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            input: src's shape:  [ser_len, batch_size,  emb_dim]
            outpou: src shape: [ser_len, batch_size,  emb_dim]
                    attns : mean attention of multi-head , shape: [batch_size, ser_len, ser_len]
        """
        # Self-atten q=k=v
        src2, attns = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)  # RestAdd
        src = self.norm1(src)  # LayerNorm

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))  # FFN
        src = src + self.dropout2(src2)  # RestAdd
        src = self.norm2(src)  # LayerNorm

        return src, attns
# ====================================================================================================
class My_TransformerEncoder(nn.Module):
    r"""TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(My_TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) \
            -> [Tensor, Tensor]:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional). 传入kvb-mask shape: [batch_size*num_head,seq_len,seq_len]
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        global attens
        output = src

        for mod in self.layers:
            output, attens = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output, attens
# ====================================================================================================

class PacketEmbedding(nn.Module):
    def __init__(self, embedding_sizes, embedding_dim=128):
        super(PacketEmbedding, self).__init__()

        emb_count = 0

        self.embedding_layers, self.num_emb = self._embedding(embedding_dim, embedding_sizes, emb_count)

    def _embedding(self, embedding_dim, embedding_sizes, emb_count):
        embedding_layers = nn.Sequential()
        embeddings = [
            nn.Embedding(size, embedding_dim) for size in embedding_sizes
        ]

        for embedding in embeddings:
            embedding_layers.add_module("embedding_layer" + str(emb_count), embedding)
            emb_count = emb_count + 1

        return embedding_layers, emb_count

    def forward(self, x):
        emb = sum([self.embedding_layers[i](x[:, :, i]) for i in range(self.num_emb)])

        return emb

# ====================================================================================================
# KVEC
class MixHaltFormer(nn.Module):
    def __init__(self, d_model, pck_embedding_sizes, nhead, num_encoder_layers, num_classes, num_substream,
                 dim_feedforward, dropout, activation, MASK_MODE, lam, bet,rnn_cell,rnn_nhid,rnn_nlayers):
        super(MixHaltFormer, self).__init__()

        ####----parameter definition----####
        self.nhead = nhead
        self.num_classes = num_classes
        self.num_substream = num_substream
        self.d_model = d_model
        self.MASK_MODE = MASK_MODE
        self.lam = lam
        self.bet = bet

        self.rnn_cell = rnn_cell
        ninp = d_model # rnn input features dims
        self.nhid = rnn_nhid
        self.nlayers = rnn_nlayers

        ####----sub-network definition----####
        self.embed = My_Embedding(pck_vocab_size=pck_embedding_sizes, d_model=d_model, n_segments=num_substream)
        encoderlayer = EncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        self.encoder = My_TransformerEncoder(encoderlayer, num_encoder_layers)
        self.RNN = torch.nn.GRU(ninp, self.nhid, self.nlayers , bidirectional=False)

        ####---- action scpace----####
        self.controller = Controller(d_model, 1)
        self.baseline = BaselineNetwork(d_model, 1)
        ####---- classifier----####
        self.classify = nn.Linear(d_model, num_classes)

    def get_kvb_mask(self, value, key, burst, lengths, num_heads, mask_mode='KVB_MASK'):
        """
        generate attention mask matrix
        :param key: shape : [batch_size,seq_length]
        :param value: shape : [batch_size,seq_length]
        :param burst: shape : [batch_size,seq_length]
        :param num_heads:
        :param lengths: a list contain flow length in batch
        :return: kvb_mask : [batch_size*num_heads,seq_len,seq_len]
                 key_mask: [batch_size,seq_len,seq_len]
                 burst_mask: [batch_size,seq_len,seq_len]
        """
        key_mask = torch.eq(key.unsqueeze(2), key.unsqueeze(1)).to(device)
        if mask_mode == 'KVB_MASK':
            value_mask = (torch.eq(value.unsqueeze(2), value.unsqueeze(1)) *
                          torch.triu(torch.ones_like(key_mask), diagonal=1).transpose(1, 2)).to(
                device)
            value_mask_wo_padding = torch.zeros_like(value_mask)
            for i in range(value.shape[0]):
                len = lengths[i]
                value_mask_wo_padding[i][:len, :len] = value_mask[i][:len, :len]

            value_mask_wo_same_key = torch.zeros_like(value_mask)
            value_mask_wo_same_key = torch.where((key_mask == False) & (value_mask_wo_padding == True),
                                                 value_mask_wo_padding, value_mask_wo_same_key)

            burst_mask = torch.zeros_like(key_mask)

            value_indx = torch.where(value_mask_wo_same_key == 1)
            value_indx = list(zip(value_indx[0].tolist(), value_indx[1].tolist(), value_indx[2].tolist()))
            for (b, x, y) in value_indx:
                burst_mask[b, x, :] |= (burst[b] == burst[b, y])

            kvb_mask = key_mask + value_mask + burst_mask

            # generate attention mask matrix for multi-head
            bsz, max_len, _ = kvb_mask.shape
            kvb_mask = kvb_mask.repeat(1, 1, num_heads).transpose(0, 1)
            multi_kvb_mask = kvb_mask.contiguous().view(max_len, bsz * num_heads, max_len).transpose(0, 1)

            return ~(multi_kvb_mask.to(device)), key_mask, burst_mask

        elif mask_mode == 'KEY_MASK':

            bsz, max_len, _ = key_mask.shape
            key_mask = key_mask.repeat(1, 1, num_heads).transpose(0, 1)
            multi_key_mask = key_mask.contiguous().view(max_len, bsz * num_heads, max_len).transpose(0, 1)

            return ~(multi_key_mask.to(device))  # just use key-mask isolate diff sub-stream

    def initHidden(self, bsz):
        """Initialize hidden states"""
        return (torch.zeros(self.nlayers, bsz, self.nhid),
                    torch.zeros(self.nlayers, bsz, self.nhid))

    def meanpoling(self,t,h_sum,h_t):
        """Mean Pooling"""
        return (h_sum+h_t)/t

    def forward(self, x, key, burst, ret_pos, value2token_offset, test=False, attn_out=False):
        '''
        :param x:  [batch_size,ser_len,fet_dim]
        :param key: [batch_size,ser_len]
        :param epoch:
        :param test:
        :param MASK_MODE: 'KV_MASK','KEY_MASK','NO'
        :return:
        '''
        global key_mask, burst_mask
        """Compute halting points and predictions"""
        if test:
            self.controller._epsilon = 0.0
        else:
            self.controller._epsilon = self._epsilon

        B, S = x.shape
        pad_mask = (x[:, :] == 0).to(device)
        lens = (S - pad_mask.sum(dim=1)).to(device)
        casual_mask = (torch.triu(torch.ones(B * self.nhead, S, S), diagonal=1) == 1).to(device)

        # MASK mode
        if self.MASK_MODE == 'KVB_MASK':
            kvb_mask, key_mask, burst_mask = self.get_kvb_mask(x, key, burst, lengths=lens.tolist(),
                                                               num_heads=self.nhead, mask_mode=self.MASK_MODE)
            att_mask = (casual_mask | kvb_mask).to(device)
        elif self.MASK_MODE == 'KEY_MASK':
            key_mask = self.get_kvb_mask(x, key, burst, lengths=lens.tolist(), num_heads=self.nhead,
                                         mask_mode=self.MASK_MODE)
            att_mask = (casual_mask | key_mask).to(device)
        else:  # 'NO_MASK'
            att_mask = casual_mask.to(device)

        # 1.Inputting Embedding
        in_emb = self.embed(x, key, ret_pos, value2token_offset)
        in_emb = in_emb.transpose(0, 1)

        # 2.Attention Layer
        enc_output, last_layer_attentions = self.encoder(in_emb, mask=att_mask)
        enc_output = torch.where(pad_mask.unsqueeze(2), torch.zeros([B, S, self.d_model]).to(device),
                                 enc_output.transpose(0, 1))

        # 3. Separate tangled flows
        rnn_bsz = B * self.num_substream
        enc_out_single = torch.zeros((rnn_bsz, 256, self.d_model)).to(device)
        halt_points = torch.zeros((rnn_bsz)).to(device)
        pad_mask = torch.zeros([rnn_bsz,S]).to(device)

        for i in range(B):
            l = lens[i]
            key_l = key[i][:l]
            s_a_idx, s_b_idx = (key_l == 0).nonzero(as_tuple=False).to(device), (key_l == 1).nonzero(as_tuple=False).to(
                device)
            l_a, l_b = s_a_idx.shape[0], s_b_idx.shape[0]
            halt_points[i], halt_points[i + B] = l_a, l_b

            enc_out_a, enc_out_b = enc_output[i][s_a_idx], enc_output[i][s_b_idx]

            enc_out_single[i, :l_a] = enc_out_a.squeeze(1)
            enc_out_single[i + B, :l_b] = enc_out_b.squeeze(1)

            pad_mask[i, :l_a] = 1
            pad_mask[i+B, :l_b] = 1

        enc_out_single = torch.transpose(enc_out_single, 0, 1)

        # 4.Policy Halting
        predictions = torch.zeros((rnn_bsz, self.num_classes)).to(device)
        halt_points = halt_points.unsqueeze(1)
        y_bar = halt_points.clone().to(device)
        batch_max_length = y_bar.max().item()

        h_0,c_0 = self.initHidden(rnn_bsz)
        h_0 = h_0.to(device)
        c_0 = c_0.to(device)

        h_sum = torch.zeros((1,rnn_bsz,self.d_model)).to(device)

        baselines = []  # baselines of halting policy
        actions = []  # actions
        log_pi = []  # Log probability of chosen actions
        halt_probs = []  # the -Log probability of halt in current stat
        logitss = []

        for t in range(S):
            cur_enc_out = enc_out_single[t].unsqueeze(0)

            rnn_output = self.meanpoling(t+1,h_sum,cur_enc_out)
            h_sum += cur_enc_out

            logits = self.classify(rnn_output.squeeze())
            a_t, p_t, w_t = self.controller(rnn_output)
            PAD = pad_mask[:, t].unsqueeze(dim=1).to(device)
            time = torch.tensor([t]).view(1, 1).repeat(rnn_bsz, 1).float().to(device)
            predictions = torch.where((a_t == 1) & (predictions == 0) & (PAD == 1), logits, predictions)
            halt_points = torch.where((halt_points > time) & (a_t == 1), time, halt_points)
            b_t = self.baseline(rnn_output)

            actions.append(a_t.squeeze())
            baselines.append(b_t.squeeze())
            log_pi.append(p_t.squeeze())
            halt_probs.append(w_t.squeeze())
            logitss.append(logits)

            if (halt_points == y_bar).sum() == 0 or i >= batch_max_length:
                break

        self.last_logits = torch.zeros((rnn_bsz, self.num_classes)).to(device)
        halt_logits = torch.zeros((B*self.num_substream, self.num_classes)).to(device)

        if (halt_points == y_bar).sum() != 0:
            for b in range(rnn_bsz):
                len = y_bar[b].item()
                last_idx = int(min(len,logitss.__len__()))
                self.last_logits[b] = logitss[last_idx -1][b]
            halt_logits = torch.where(predictions == 0.0, self.last_logits, predictions)
        else:
            halt_logits = predictions

        self.actions = torch.stack(actions).transpose(0, 1)
        self.baselines = torch.stack(baselines).transpose(0, 1)
        self.log_pi = torch.stack(log_pi).transpose(0, 1)
        self.halt_probs = torch.stack(halt_probs).transpose(0, 1)

        self.grad_mask = torch.zeros_like(self.actions)
        for b in range(rnn_bsz):
            self.grad_mask[b, :(1 + halt_points[b, 0]).long()] = 1

        if attn_out:
            return halt_logits, halt_points, lens, y_bar, last_layer_attentions, key_mask, burst_mask
            # output last_layer_attentions
        else:
            return halt_logits, halt_points, lens, y_bar

    def computeLoss(self, logits, y):
        rnn_bsz, num_classes = logits.shape

        # --- compute reward ---.
        _, y_hat = torch.max(torch.softmax(logits, dim=1), dim=1)
        y = torch.cat((y[:, 0], y[:, 1]), dim=0)

        self.r = (2 * (y_hat == y).float() - 1).detach().unsqueeze(1)  # r.shape:[rnn_bsz,1]
        self.R = torch.zeros_like(self.grad_mask)
        self.R = self.r * self.grad_mask

        # --- rescale reward with baseline ---
        b = self.grad_mask * self.baselines
        self.adjusted_reward = self.R - b.detach()

        # --- compute losses ---
        MSE = torch.nn.MSELoss()
        CE = torch.nn.CrossEntropyLoss()
        self.loss_b = MSE(b, self.R)

        self.loss_r = (-self.log_pi * self.adjusted_reward).sum() / (self.grad_mask.sum() / rnn_bsz)

        self.loss_c = CE(logits,y)  # Classification loss
        self.wait_penalty = (self.halt_probs * self.grad_mask).sum(1).mean()  # Penalize late predictions
        loss = self.bet * (self.loss_r) + self.loss_b + self.loss_c + self.lam * (self.wait_penalty)

        return loss, self.loss_c, self.loss_r, self.loss_b, self.wait_penalty
# ====================================================================================================
# KVEC
class MixHaltFormer2(nn.Module):
    def __init__(self, d_model, pck_embedding_sizes, nhead, num_encoder_layers, num_classes, num_substream,
                 dim_feedforward, dropout, activation, MASK_MODE, lam, bet,rnn_cell,rnn_nhid,rnn_nlayers):
        super(MixHaltFormer2, self).__init__()

        ####----parameter definition----####
        self.nhead = nhead
        self.num_classes = num_classes
        self.num_substream = num_substream
        self.d_model = d_model
        self.MASK_MODE = MASK_MODE
        self.lam = lam
        self.bet = bet

        ###----parameter for rnn----####
        self.rnn_cell = rnn_cell
        ninp = d_model
        self.nhid = rnn_nhid
        self.nlayers = rnn_nlayers

        ####----sub-network definition----####
        self.embed = My_Embedding(pck_vocab_size=pck_embedding_sizes, d_model=d_model, n_segments=num_substream)
        encoderlayer = EncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
        self.encoder = My_TransformerEncoder(encoderlayer, num_encoder_layers)

        self.RNN = torch.nn.LSTM(ninp, self.nhid, self.nlayers)

        ####---- action scpace----####
        self.controller = Controller(d_model, 1)
        self.baseline = BaselineNetwork(d_model, 1)

        self.classify = nn.Linear(d_model, num_classes)

    def get_kvb_mask(self, value, key, burst, lengths, num_heads, mask_mode='KVB_MASK'):
        """
        :param key: shape : [batch_size,seq_length]
        :param value: shape : [batch_size,seq_length]
        :param burst: shape : [batch_size,seq_length]
        :param num_heads:
        :param lengths: a list contain flow length in batch
        :return: kvb_mask : [batch_size*num_heads,seq_len,seq_len] 广播到multi-head
                 key_mask: [batch_size,seq_len,seq_len] 为True的地方为key-correlation-->inner-attn的范围
                 burst_mask: [batch_size,seq_len,seq_len] 为True的地方为value-correlation-->inter-attn的范围
        """
        key_mask = torch.eq(key.unsqueeze(2), key.unsqueeze(1)).to(device)
        if mask_mode == 'KVB_MASK':
            value_mask = (torch.eq(value.unsqueeze(2), value.unsqueeze(1)) *
                          torch.triu(torch.ones_like(key_mask), diagonal=1).transpose(1, 2)).to(
                device)
            value_mask_wo_padding = torch.zeros_like(value_mask)
            for i in range(value.shape[0]):
                len = lengths[i]
                value_mask_wo_padding[i][:len, :len] = value_mask[i][:len, :len]

            value_mask_wo_same_key = torch.zeros_like(value_mask)
            value_mask_wo_same_key = torch.where((key_mask == False) & (value_mask_wo_padding == True),
                                                 value_mask_wo_padding, value_mask_wo_same_key)
            burst_mask = torch.zeros_like(key_mask)

            value_indx = torch.where(value_mask_wo_same_key == 1)
            value_indx = list(zip(value_indx[0].tolist(), value_indx[1].tolist(), value_indx[2].tolist()))
            for (b, x, y) in value_indx:
                burst_mask[b, x, :] |= (burst[b] == burst[b, y])

            kvb_mask = key_mask + value_mask + burst_mask
            bsz, max_len, _ = kvb_mask.shape
            kvb_mask = kvb_mask.repeat(1, 1, num_heads).transpose(0, 1)
            multi_kvb_mask = kvb_mask.contiguous().view(max_len, bsz * num_heads, max_len).transpose(0, 1)

            return ~(multi_kvb_mask.to(device)), key_mask, burst_mask

        elif mask_mode == 'KEY_MASK':
            bsz, max_len, _ = key_mask.shape
            key_mask = key_mask.repeat(1, 1, num_heads).transpose(0, 1)
            multi_key_mask = key_mask.contiguous().view(max_len, bsz * num_heads, max_len).transpose(0, 1)

            return ~(multi_key_mask.to(device))

    def initHidden(self, bsz):
        """Initialize hidden states"""
        return (torch.zeros(self.nlayers, bsz, self.nhid),
                    torch.zeros(self.nlayers, bsz, self.nhid))

    def meanpoling(self,t,h_sum,h_t):
        """Mean Pooling"""
        return (h_sum+h_t)/t

    def forward(self, x, key, burst, ret_pos, value2token_offset, test=False, attn_out=False):
        '''
        :param x:  [batch_size,ser_len,fet_dim]
        :param key: [batch_size,ser_len]
        :param epoch:
        :param test:
        :param MASK_MODE: 'KV_MASK','KEY_MASK','NO'
        :return:
        '''
        global key_mask, burst_mask
        """Compute halting points and predictions"""
        if test:
            self.controller._epsilon = 0.0
        else:
            self.controller._epsilon = self._epsilon

        B, S = x.shape
        pad_mask = (x[:, :] == 0).to(device)
        lens = (S - pad_mask.sum(dim=1)).to(device)
        casual_mask = (torch.triu(torch.ones(B * self.nhead, S, S), diagonal=1) == 1).to(device)

        # MASK mode
        if self.MASK_MODE == 'KVB_MASK':
            kvb_mask, key_mask, burst_mask = self.get_kvb_mask(x, key, burst, lengths=lens.tolist(),
                                                               num_heads=self.nhead, mask_mode=self.MASK_MODE)
            att_mask = (casual_mask | kvb_mask).to(device)
        elif self.MASK_MODE == 'KEY_MASK':
            key_mask = self.get_kvb_mask(x, key, burst, lengths=lens.tolist(), num_heads=self.nhead,
                                         mask_mode=self.MASK_MODE)
            att_mask = (casual_mask | key_mask).to(device)

        else:  # 'NO_MASK'
            att_mask = casual_mask.to(device)
        # 1.Inputting Embedding
        in_emb = self.embed(x, key, ret_pos, value2token_offset)  # [batch_size,ser_len,emb_dim]
        in_emb = in_emb.transpose(0, 1)  # [batch_size,ser_len,emb_dim]-->[ser_len,batch_size,emb_dim]

        # 2.Attention Layer
        enc_output, last_layer_attentions = self.encoder(in_emb, mask=att_mask)
        enc_output = torch.where(pad_mask.unsqueeze(2), torch.zeros([B, S, self.d_model]).to(device),
                                 enc_output.transpose(0, 1))

        # 3.
        ##############
        rnn_bsz = B * self.num_substream
        enc_out_single = torch.zeros((rnn_bsz, 256, self.d_model)).to(device)
        sl_lens = torch.zeros((rnn_bsz)).to(device)

        sl_pad_mask = torch.zeros([rnn_bsz,S]).to(device)

        for i in range(B):
            l = lens[i]
            key_l = key[i][:l]
            s_a_idx, s_b_idx = (key_l == 0).nonzero(as_tuple=False).to(device), (key_l == 1).nonzero(as_tuple=False).to(
                device)
            l_a, l_b = s_a_idx.shape[0], s_b_idx.shape[0]
            sl_lens[i], sl_lens[i + B] = l_a, l_b

            enc_out_a, enc_out_b = enc_output[i][s_a_idx], enc_output[i][s_b_idx]

            enc_out_single[i, :l_a] = enc_out_a.squeeze(1)
            enc_out_single[i + B, :l_b] = enc_out_b.squeeze(1)

            sl_pad_mask[i, :l_a] = 1
            sl_pad_mask[i+B, :l_b] = 1

        enc_out_single = torch.transpose(enc_out_single, 0, 1)
        batch_max_length = sl_lens.max().item()

        h_0,c_0 = self.initHidden(rnn_bsz)
        h_0 = h_0.to(device)
        c_0 = c_0.to(device)
        rnn_out_single = torch.zeros_like(enc_out_single).to(device)

        for t in range(S):
            cur_enc_out = enc_out_single[t].unsqueeze(0)

            rnn_output, (h_0,c_0) = self.RNN(cur_enc_out,(h_0,c_0))
            rnn_out_single[t] = rnn_output

            if i >= batch_max_length:
                break

        enc_out_mean = torch.zeros_like(enc_output).float().to(device)
        halt_points = torch.zeros((B, self.num_substream)).to(device)
        rnn_out_single = torch.transpose(rnn_out_single, 0, 1)
        sl_lens = sl_lens.int()
        for i in range(B):
            l = lens[i]
            key_l = key[i][:l]
            s_a_idx, s_b_idx = (key_l == 0).nonzero(as_tuple=False).to(device), (key_l == 1).nonzero(as_tuple=False).to(device)
            halt_points[i] = torch.cat([s_a_idx[-1],s_b_idx[-1]])
            enc_out_mean[i,s_a_idx] = rnn_out_single[i][:sl_lens[i].item()].unsqueeze(1)
            enc_out_mean[i,s_b_idx] = rnn_out_single[i+B][:sl_lens[i+B].item()].unsqueeze(1)
        ########
        predictions = torch.zeros((B, self.num_substream,self.num_classes)).to(device) ####
        # predictions.shape=[batch_size ,num_substream, num_classes]

        halt_points = halt_points.unsqueeze(2)
        y_bar = halt_points.clone().to(device)

        baselines = []
        actions = []
        log_pi = []
        halt_probs = []
        logitss = []

        for t in range(S):
            cur_enc_out = enc_out_mean[:,t,:]
            cur_key = key[:,t].unsqueeze(1)
            logits = self.classify(cur_enc_out) #shape: [batch_size,num_classes]
            a_t, p_t, w_t = self.controller(cur_enc_out)
            PAD = pad_mask[:,t].unsqueeze(dim=1).to(device)

            predictions[:,0] = torch.where((a_t == 1) & (predictions[:,0] == 0) & (~PAD) & (cur_key==0), logits, predictions[:,0])
            predictions[:,1] = torch.where((a_t == 1) & (predictions[:,1] == 0) & (~PAD) & (cur_key==1), logits, predictions[:,1])

            time = torch.tensor([t]).view(1, 1).repeat(B, 1).float().to(device)

            # halt_points.shape:[batch_size,num_substream,1]
            halt_points[:,0] = torch.where((halt_points[:,0] > time) & (a_t == 1) & (cur_key==0), time, halt_points[:,0])
            halt_points[:,1] = torch.where((halt_points[:,1] > time) & (a_t == 1) & (cur_key==1), time, halt_points[:,1])

            b_t = self.baseline(cur_enc_out)  # shape:[batch_size , 1]

            actions.append(a_t.squeeze())
            baselines.append(b_t.squeeze())
            log_pi.append(p_t.squeeze())
            halt_probs.append(w_t.squeeze())
            logitss.append(logits)

            if (halt_points == y_bar).sum() == 0:
                break

        self.last_logits = torch.zeros((B, self.num_substream,self.num_classes)).to(device)  # shape:[batch_size,num_substream,nclasses]
        halt_logits = torch.zeros((B, self.num_substream,self.num_classes)).to(device)

        if (halt_points == y_bar).sum() != 0:
            for b in range(B):
                l_a,l_b = y_bar[b]
                last_a_idx,last_b_idx = min(int(l_a), logitss.__len__()),min(int(l_b), logitss.__len__())
                self.last_logits[b,0] = logitss[last_a_idx - 1][b]
                self.last_logits[b,1] = logitss[last_b_idx - 1][b]

            halt_logits = torch.where(predictions == 0.0, self.last_logits, predictions)
        else :
            halt_logits = predictions

        self.actions = torch.stack(actions).transpose(0, 1)
        self.baselines = torch.stack(baselines).transpose(0, 1)
        self.log_pi = torch.stack(log_pi).transpose(0, 1)
        self.halt_probs = torch.stack(halt_probs).transpose(0, 1)

        # --- Compute mask for where actions are updated ---
        self.grad_mask = torch.zeros_like(self.actions)
        grad_num = torch.arange(0, self.actions.shape[1]).repeat(B, 1).to(device)
        self.grad_key = key[:, :self.actions.shape[1]]
        self.grad_mask = torch.where(((self.grad_key == 0) & (grad_num <= halt_points[:, 0])), torch.ones_like(self.actions),self.grad_mask)
        self.grad_mask = torch.where(((self.grad_key == 1) & (grad_num <= halt_points[:, 1])), torch.ones_like(self.actions),self.grad_mask)

        if attn_out:
            return halt_logits,halt_points,lens,y_bar,last_layer_attentions,key_mask,burst_mask
            # output last_layer_attentions
        else:
            return halt_logits, halt_points, lens, y_bar

    def computeLoss(self,logits,y):
        batch_size,num_substream,num_classes = logits.shape

        # --- compute reward ---.
        _, y_hat = torch.max(torch.softmax(logits, dim=2), dim=2)

        self.r = (2 * (y_hat.float().round() == y.float()).float() - 1).detach()

        self.R = torch.zeros_like(self.grad_mask)
        self.R = torch.where((self.grad_key == 0), self.r[:, 0].unsqueeze(1) * self.grad_mask, self.R)
        self.R = torch.where((self.grad_key == 1), self.r[:, 1].unsqueeze(1) * self.grad_mask, self.R)

        # --- rescale reward with baseline ---
        b = self.grad_mask * self.baselines
        self.adjusted_reward = self.R - b.detach()

        # --- compute losses ---
        MSE = torch.nn.MSELoss()
        CE = torch.nn.CrossEntropyLoss()

        self.loss_b = MSE(b, self.R) # Baseline should approximate mean reward
        self.loss_r = (-self.log_pi * self.adjusted_reward).sum() / (self.grad_mask.sum()/batch_size)
        self.loss_c = CE(logits.reshape(batch_size*num_substream,-1), y.reshape(batch_size*num_substream)) # Classification loss
        self.wait_penalty = (self.halt_probs*self.grad_mask).sum(1).mean() # Penalize late predictions
        loss = self.bet*(self.loss_r) + self.loss_b + self.loss_c + self.lam*(self.wait_penalty)

        return loss,self.loss_c,self.loss_r,self.loss_b,self.wait_penalty
# ====================================================================================================
def exponentialDecay(N):
    tau = 1
    tmax = 4
    t = np.linspace(0, tmax, N)
    y = np.exp(-t / tau)
    y = torch.FloatTensor(y)
    return y / 10.
