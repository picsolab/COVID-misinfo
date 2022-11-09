import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (seq_ln, batch_sz, emb_sz)
        # adding positional encoding to x tensor
        # emb_sz == d_model
        
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class AttnEncoder(nn.Module):
    
    def __init__(self, emb_sz, dim_feedforward, nhead, nlayer, dropout):
        super(AttnEncoder, self).__init__()
        self.position_encoder = PositionalEncoding(emb_sz, dropout)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(emb_sz, nhead, dim_feedforward, dropout), nlayer)
        
    def forward(self, src, src_padding_mask):
        # src: (batch_sz, seq_ln, emb_sz) where `seq_ln` indicates time steps
        # src_padding_mask: (batch_sz, seq_ln)
        # hidden: (batch_sz, seq_ln, emb_sz)
        
        # print(src.shape)
        src = src.permute(1,0,2) # (seq_ln, batch_sz, emb_sz)
        src = self.position_encoder(src)
        hidden = self.transformer_encoder(src=src, src_key_padding_mask=src_padding_mask)
        hidden = hidden.permute(1,0,2) # (batch_sz, seq_ln, emb_sz)
        return hidden

class XTransformer1d(nn.Module):
    
    # use transformer encoder to encode inputs of (batch_sz, seq_ln, emb_sz)
    # we all it 1d because embedding dimensions can be taken as in_channels
    # and the sequence length is taken as 1d sequence
    
    # this module calculate the contribution tensor contrib[t] before final fusion layers
    # alpha[t] * emb[t] whose shape is (batch_sz, seq_ln, emb_sz) without summarizing embedding dimension
    
    def __init__(self, emb_sz, dim_feedforward, nhead, nlayer, dropout):
        super(XTransformer1d, self).__init__()
        
        self.encoder = AttnEncoder(emb_sz, dim_feedforward, nhead, nlayer, dropout)
        self.ln_alpha = nn.Linear(emb_sz, 1)
        
    def masker(self, inp):
        return inp.eq(1)
    
    def forward(self, inp_emb, inp):
        
        padding_mask = self.masker(inp) # (batch_sz, seq_ln)
        embedding = inp_emb.masked_fill(padding_mask.unsqueeze(-1)==True, 0) # (batch_sz, seq_ln, emb_sz)
    
        hidden = self.encoder(embedding, padding_mask) # (batch_sz, seq_ln, emb_sz)
        alpha = F.softmax(self.ln_alpha(hidden).squeeze(), dim=-1).unsqueeze(-1) # (batch_sz, seq_ln, 1)
        
        contrib = embedding.mul(alpha) # (batch_sz, seq_ln, emb_sz)
        
        return contrib

class XTransformer2d(nn.Module):
    
    # use transformer encoder to encode inputs of (batch_sz, seq_ln, token_cnt, emb_sz)
    # we all it 2d because embedding dimensions (emb_sz) can be taken as in_channels
    # and the sequence length (token_cnt) is taken as height (width)
    
    # this module calculate the contribution tensor before final fusion layers
    # alpha[t] * beta[t] * emb[t,i] whose shape is (batch_sz, seq_ln, token_cnt, emb_sz)
    # note: beta's 3rd dimension is not equal to token_cnt, instead its shape is equal to
    # hidden's shape, i.e., (batch_sz, seq_ln, emb_sz)
    # eqivalently, beta is attention allocated on each unit of latent space
    
    def __init__(self, emb_sz, dim_feedforward, nhead, nlayer, dropout):
        super(XTransformer2d, self).__init__()

        self.encoder = AttnEncoder(emb_sz, dim_feedforward, nhead, nlayer, dropout)
        self.ln_beta = nn.Linear(emb_sz, emb_sz)
        self.ln_alpha = nn.Linear(emb_sz, 1)
    
    def masker(self, inp):
        return inp.eq(1), inp.eq(1).all(dim=-1).squeeze()
    
    def forward(self, inp_emb, inp):
        
        token_padding_mask, timestep_padding_mask = self.masker(inp)
        embedding = inp_emb.masked_fill(token_padding_mask.unsqueeze(-1)==True, 0) # (batch_sz, seq_ln, token_cnt, emb_sz)
        
        hidden = self.encoder(embedding.sum(dim=2), timestep_padding_mask) # (batch_sz, seq_ln, emb_sz)

        alpha = F.softmax(self.ln_alpha(hidden).squeeze(), dim=-1).unsqueeze(-1).unsqueeze(-1) # (batch_sz, seq_ln, 1, 1)
        beta = torch.tanh(self.ln_beta(hidden)).unsqueeze(2) # (batch_sz, seq_ln, 1, emb_sz)
        
        contrib = embedding.mul(beta).mul(alpha) # (batch_sz, seq_ln, token_cnt, emb_sz)
        
        return contrib

class Unified_xtransformer1d(nn.Module):
    def __init__(self, ntoken, dim_feedforward, nhead, nlayer, dropout, emb_sz, nclass, verbose):
        
        super(Unified_xtransformer1d, self).__init__()
        
        self.emb_sz = emb_sz
        
        self.emb_layer = nn.Embedding(ntoken, emb_sz)
        
        if verbose != 'both':
            self.xtransformer_layers = nn.ModuleList([
                XTransformer1d(emb_sz, dim_feedforward, nhead, nlayer, dropout)
            ])
            self.final_layer = nn.Linear(emb_sz, nclass)
        else:
            self.xtransformer_layers = nn.ModuleList([
                XTransformer1d(emb_sz, dim_feedforward, nhead, nlayer, dropout),
                XTransformer1d(emb_sz, dim_feedforward, nhead, nlayer, dropout)
            ])
            self.final_layer = nn.Linear(emb_sz * 2, nclass)
        
        self.verbose = verbose
    
    def lookup_embeddings(self):
        return self.emb_layer.weight
    
    def forward(self, inputs):
        
        if self.verbose == 'ego':
            inputs = (inputs[0], )
        elif self.verbose == 'ngh':
            inputs = (inputs[1], )
            
        embeddings = [self.emb_layer(inputs[i]) for i in range(len(inputs))]
        
        contributions = []
        for i in range(len(inputs)):
            module = self.xtransformer_layers[i]
            contributions.append(module(embeddings[i], inputs[i]))
        
        uservecs = [cb.sum(dim=1) for cb in contributions]
        context = torch.cat(uservecs, 1) # (batch_sz, 2, emb_sz)
        
        # context = uservecs.view(-1, 2 * self.emb_sz)
        logits = self.final_layer(context) # (batch_sz, nclass)
        
        
#         W = self.final_layer.weight.transpose(0,1) # (emb_sz * 2, nclass)
#         contributions = [contributions[0].matmul(W[0:self.emb_sz,:]),
#                          contributions[1].matmul(W[self.emb_sz:(self.emb_sz * 2),:])]
        
        return logits # , uservecs, contributions
        
class Unified_xtransformerXd(nn.Module):
    
    def __init__(self, ntokens, dim_feedforwards, nheads, nlayers, dropout_, emb_sz_, nclass_):
        
        # unify 4 sources of inputs: two 2d text inputs and another two 1d netloc inputs
        # ntokens is a tuple of two elements, the other args contain four elements
        
        super(Unified_xtransformerXd, self).__init__()
        
        self.emb_sz = emb_sz_
        
        self.emb_layers = [nn.Embedding(ntoken, emb_sz_) for ntoken in ntokens]
        
        self.n_sources = len(dim_feedforwards)
        if self.n_sources == 4:
            self.xtransformer_layers = nn.ModuleList([
                XTransformer1d(emb_sz_, dim_feedforwards[0], nheads[0], nlayers[0], dropout_),
                XTransformer1d(emb_sz_, dim_feedforwards[1], nheads[1], nlayers[1], dropout_),
                XTransformer2d(emb_sz_, dim_feedforwards[2], nheads[2], nlayers[2], dropout_),
                XTransformer2d(emb_sz_, dim_feedforwards[3], nheads[3], nlayers[3], dropout_)
            ])
        elif self.n_sources == 2:
            self.xtransformer_layers = nn.ModuleList([
                XTransformer1d(emb_sz_, dim_feedforwards[0], nheads[0], nlayers[0], dropout_),
                XTransformer2d(emb_sz_, dim_feedforwards[1], nheads[1], nlayers[1], dropout_),
            ])
        else:
            raise ValueError('invalid length!')
        
        # self.ln_gamma = nn.Linear(emb_sz_, 1)
        self.final_layer = nn.Linear(emb_sz_ * self.n_sources, nclass_)
    
    def forward(self,inputs):
        
        # inputs is a tuple of four elements containing ego's netloc, ngh's netloc, ego's text and ngh's text
        # netloc shape is (batch_sz, seq_ln), text shape is (batch_sz, seq_ln, token_cnt)
        

        ### === EMBEDDING === ###
        if self.n_sources == 4:
            embeddings = [self.emb_layers[int(i/2)](inputs[i]) for i in range(4)]
        elif self.n_sources == 2:
            inputs = (inputs[0], inputs[2])
            embeddings = [self.emb_layers[i](inputs[i]) for i in range(2)]
        
        ### === GET CONTRIB TENSORS === ###
        contributions = []
        for i in range(self.n_sources):
            module = self.xtransformer_layers[i]
            contributions.append(module(embeddings[i], inputs[i]))
            # (batch_sz, seq_ln, emb_sz)
            # (batch_sz, seq_ln, token_cnt, emb_sz)
        
        
        ### === USER VECTORS === ###
        uservecs = []
        for i in range(self.n_sources):
            if len(contributions[i].shape) == 3:
                uservecs.append(contributions[i].sum(dim=1).unsqueeze(1))
            else:
                uservecs.append(contributions[i].sum(dim=1).sum(dim=1).unsqueeze(1))
        uservecs = torch.cat(uservecs, 1) # (batch_sz, 4, emb_sz)
        
        ### === FUSION LAYER === ###
        
        # calculate the coefficients of four input sources
        # gamma = F.softmax(self.ln_gamma(uservecs), dim=1) # (batch_sz, 4, 1)
        
        # context = uservecs.mul(gamma) # (batch_sz, 4, emb_sz)
        # context = context.sum(dim=1) # (batch_sz, emb_sz)
        
        context = uservecs.view(-1, self.emb_sz * self.n_sources)
        
        logits = self.final_layer(context) # (batch_sz, nclass)
        
        ### === UPDATE CONTRIBUTION === ###
        W = self.final_layer.weight.transpose(0,1) # (emb_sz, nclass)
#         for i in range(self.n_sources):
#             if len(contributions[i].shape) == 3:
#                 contributions[i] = contributions[i].matmul(W).mul(gamma[:,i,:].unsqueeze(-1)) # (batch_sz, seq_ln, nclass)
#             else:
#                 contributions[i] = contributions[i].matmul(W).mul(
#                     gamma[:,i,:].unsqueeze(-1).unsqueeze(-1)) # (batch_sz, seq_ln, token_cnt, nclass)

        for i in range(self.n_sources):
            contributions[i] = contributions[i].matmul(W[(i * self.emb_sz):((i + 1) * self.emb_sz)])
        
        return logits, uservecs, contributions

class xtransformer1d(nn.Module):
    
    def __init__(self, ntoken, emb_sz, dim_feedforward, nhead, nlayer, nclass, dropout):
        super(xtransformer1d, self).__init__()
        self.emb_layer = nn.Embedding(ntoken, emb_sz)
        self.encoder = AttnEncoder(emb_sz, dim_feedforward, nhead, nlayer, dropout)
        
        self.linear_alpha = nn.Linear(emb_sz, 1)
        self.final_layer = nn.Linear(emb_sz, nclass)
    
    def masker(self, inp):
        return inp.eq(1)
    
    def lookup_embeddings(self):
        return self.emb_layer.weight
    
    def forward(self, inp):
        # inp: (batch_sz, seq_ln)
        # returns 3 tensors
        # prob (batch_sz, nclass)
        # uservec (batch_sz, emb_sz)
        # contrib (batch_sz, seq_ln, token_cnt, nclass)
        
        inp = inp[0]
        padding_mask = self.masker(inp) # (batch_sz, seq_ln)
        embedding = self.emb_layer(inp).masked_fill(
            padding_mask.unsqueeze(-1)==True, 0) # (batch_sz, seq_ln, emb_sz)
    
        ## next we start calculating coefficient "alpha" for time steps
        hidden = self.encoder(embedding, padding_mask) # (batch_sz, seq_ln, emb_sz)
        alpha = F.softmax(
            self.linear_alpha(hidden).squeeze(), dim=-1).unsqueeze(-1) # (batch_sz, seq_ln, 1)
        
        ## next we calculate contributions via multiplying embeddings and coefficients 
        ## (i.e., element-wise multiplication for 3d tensor)
        contrib = embedding.mul(alpha) # (batch_sz, seq_ln, emb_sz)
        
        # sum along time dimension to get user vector (or context vector)
        uservec = contrib.sum(dim=1) # (batch_sz, emb_sz)

        # prob = F.softmax(self.final_layer(uservec), dim=-1) # (batch_sz, nclass)
        logits = self.final_layer(uservec) # (batch_sz, nclass)
        
        # contrib is a 3d tensor indicating which domain contribute to which class
        # for <pad> the element in this tensor would be zero
        contrib = contrib.matmul(self.final_layer.weight.transpose(0,1)) # (batch_sz, seq_ln, nclass)
        return logits, uservec, contrib

class xtransformer2d(nn.Module):
    
    def __init__(self, ntoken, emb_sz, dim_feedforward, nhead, nlayer, nclass, dropout):
        super(xtransformer2d, self).__init__()
        self.emb_layer = nn.Embedding(ntoken, emb_sz)
        self.encoder = AttnEncoder(emb_sz, dim_feedforward, nhead, nlayer, dropout)
        
        self.linear_beta = nn.Linear(emb_sz, emb_sz)
        self.linear_alpha = nn.Linear(emb_sz, 1)
        self.final_layer = nn.Linear(emb_sz, nclass)
    
    def masker(self, inp):
        return inp.eq(1), inp.eq(1).all(dim=-1).squeeze()
    
    def forward(self, inp):
        # inp: (batch_sz, seq_ln, token_cnt)
        # returns 3 tensors
        # prob (batch_sz, nclass)
        # uservec (batch_sz, emb_sz)
        # contrib (batch_sz, seq_ln, token_cnt, nclass)
        
        inp = inp[2]
        
        token_padding_mask, timestep_padding_mask = self.masker(inp)
        # (batch_sz, seq_ln, token_cnt)
        # (batch_sz, seq_ln)
        
        embedding = self.emb_layer(inp).masked_fill(
            token_padding_mask.unsqueeze(-1)==True, 0) # (batch_sz, seq_ln, token_cnt, emb_sz)
        
        ## next we start learning coefficients "alpha" and "beta"
        agg_emb = embedding.sum(dim=2) # aggregate along `token_cnt` (batch_sz, seq_ln, emb_sz)
        hidden = self.encoder(agg_emb, timestep_padding_mask) # (batch_sz, seq_ln, emb_sz)
        # alpha indicates coefficients of time steps
        alpha = F.softmax(
            self.linear_alpha(hidden).squeeze(), dim=-1).unsqueeze(-1).unsqueeze(-1) # (batch_sz, seq_ln, 1, 1)
        # beta indicates coefficients of embedding dimensions
        beta = torch.tanh(self.linear_beta(hidden)).unsqueeze(2) # (batch_sz, seq_ln, 1, emb_sz)
        
        ## next we calculate contributions via multiplying embeddings and coefficients 
        ## (element-wise multiplication for 4d tensor)
        contrib = embedding.mul(beta).mul(alpha) # (batch_sz, seq_ln, token_cnt, emb_sz)
        
        # sum along time and token dimension to get user vector (also called context vector)
        uservec = contrib.sum(dim=1).sum(dim=1) # (batch_sz, emb_sz)

        # prob = F.softmax(self.final_layer(uservec), dim=-1) # (batch_sz, nclass)
        logits = self.final_layer(uservec) # (batch_sz, nclass)
        
        # contrib is a 4d tensor indicating which token at which timestep contribute to which class
        # for <pad> the element in this tensor would be zero
        contrib = contrib.matmul(self.final_layer.weight.transpose(0,1)) # (batch_sz, seq_ln, token_cnt, nclass)
        return logits, uservec, contrib