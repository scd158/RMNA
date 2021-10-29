import numpy as np
import torch
import torch.nn as nn
import time
import torch.nn.functional as F
from torch.autograd import Variable


CUDA = torch.cuda.is_available()


class ConvKB(nn.Module):
    def __init__(self, input_dim, input_seq_len, in_channels, out_channels, drop_prob, alpha_leaky):
        super().__init__()

        self.conv_layer = nn.Conv2d(
            in_channels, out_channels, (1, input_seq_len))  # kernel size -> 1*input_seq_length(i.e. 2)
        self.dropout = nn.Dropout(drop_prob)
        self.non_linearity = nn.ReLU()
        self.fc_layer = nn.Linear((input_dim) * out_channels, 1)

        nn.init.xavier_uniform_(self.fc_layer.weight, gain=1.414)
        nn.init.xavier_uniform_(self.conv_layer.weight, gain=1.414)

    def forward(self, conv_input):

        batch_size, length, dim = conv_input.size()
        # assuming inputs are of the form ->
        conv_input = conv_input.transpose(1, 2)
        # batch * length(which is 3 here -> entity,relation,entity) * dim
        # To make tensor of size 4, where second dim is for input channels
        conv_input = conv_input.unsqueeze(1)

        out_conv = self.dropout(
            self.non_linearity(self.conv_layer(conv_input)))

        input_fc = out_conv.squeeze(-1).view(batch_size, -1)
        output = self.fc_layer(input_fc)
        return output


class SpecialSpmmFunctionFinal(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, edge, edge_w, N, E, out_features):
        # assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(
            edge, edge_w, torch.Size([N, N, out_features]))
        b = torch.sparse.sum(a, dim=1)
        ctx.N = b.shape[0]
        ctx.outfeat = b.shape[1]
        ctx.E = E
        ctx.indices = a._indices()[0, :]
        return b.to_dense()

    @staticmethod
    def backward(ctx, grad_output):
        grad_values = None
        if ctx.needs_input_grad[1]:
            edge_sources = ctx.indices

            if(CUDA):
                edge_sources = edge_sources.cuda()

            grad_values = grad_output[edge_sources]
            # grad_values = grad_values.view(ctx.E, ctx.outfeat)
            # print("Grad Outputs-> ", grad_output)
            # print("Grad values-> ", grad_values)
        return None, grad_values, None, None, None


class SpecialSpmmFinal(nn.Module):
    def forward(self, edge, edge_w, N, E, out_features):
        return SpecialSpmmFunctionFinal.apply(edge, edge_w, N, E, out_features)


class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, num_nodes, in_features, out_features, nrela_dim, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()#N, 200, 200, 200, 
        self.in_features = in_features
        self.out_features = out_features
        self.num_nodes = num_nodes
        self.alpha = alpha
        self.concat = concat
        self.nrela_dim = nrela_dim

        self.a = nn.Parameter(torch.zeros(
            size=(out_features, 2 * in_features + nrela_dim)))
        self.new_a = nn.Parameter(torch.zeros(
            size=(out_features, 2 * in_features + nrela_dim + 4)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)
        nn.init.xavier_normal_(self.new_a.data, gain=1.414)
        self.a_2 = nn.Parameter(torch.zeros(size=(1, out_features)))
        self.new_a_2 = nn.Parameter(torch.zeros(size=(1, out_features)))
        nn.init.xavier_normal_(self.a_2.data, gain=1.414)
        nn.init.xavier_normal_(self.new_a_2.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm_final = SpecialSpmmFinal()

    def forward(self, input, edge, new_edge, edge_embed, new_edge_embed, new_edge_other, edge_list_nhop, edge_embed_nhop):
        N = input.size()[0]

        # Self-attention on the nodes - Shared attention mechanism
        if(edge_list_nhop.size()[0] != 0):
            edge = torch.cat((edge[:, :], edge_list_nhop[:, :]), dim=1)
            edge_embed = torch.cat(
                (edge_embed[:, :], edge_embed_nhop[:, :]), dim=0)

        edge_h = torch.cat((input[edge[0, :], :], input[edge[1, :], :], edge_embed[:, :]), dim = 1).t() #3.19 to be format
		
        new_edge_h = torch.cat((input[new_edge[0, :], :], input[new_edge[1, :], :], new_edge_embed[:, :], new_edge_other.t()), dim=1).t()
        # edge_h: (2*in_dim + nrela_dim) x E
		# new_edge_h: (2*in_dim + nrela_dim + 4) x E

        edge_m = self.a.mm(edge_h)
        new_edge_m = self.new_a.mm(new_edge_h)
        # edge_m: D * E
		# edge_m: D * E

        # to be checked later
        powers = -self.leakyrelu(self.a_2.mm(edge_m).squeeze())
        new_powers = -self.leakyrelu(self.new_a_2.mm(new_edge_m).squeeze())
		
        edge_e = torch.exp(powers).unsqueeze(1)
        new_edge_e = torch.exp(new_powers).unsqueeze(1)
        assert not torch.isnan(edge_e).any()
        assert not torch.isnan(new_edge_e).any()#3.18 end
        # edge_e: E
		# new_edge_e: E

        e_rowsum = self.special_spmm_final(
            edge, edge_e, N, edge_e.shape[0], 1)
        new_e_rowsum = self.special_spmm_final(
            new_edge, new_edge_e, N, new_edge_e.shape[0], 1)

        e_rowsum[e_rowsum == 0.0] = 1e-12
        new_e_rowsum[new_e_rowsum == 0.0] = 1e-12

        e_rowsum = e_rowsum
        new_e_rowsum = new_e_rowsum
        # e_rowsum: N x 1
        edge_e = edge_e.squeeze(1)
        new_edge_e = new_edge_e.squeeze(1)

        edge_e = self.dropout(edge_e)
        new_edge_e = self.dropout(new_edge_e)
        # edge_e: E

        edge_w = (edge_e * edge_m).t()
        new_edge_w = (new_edge_e * new_edge_m).t()
        # edge_w: E * D
        h_prime = self.special_spmm_final(
            edge, edge_w, N, edge_w.shape[0], self.out_features)

        new_h_prime = self.special_spmm_final(
            new_edge, new_edge_w, N, new_edge_w.shape[0], self.out_features)
        assert not torch.isnan(h_prime).any()
        assert not torch.isnan(new_h_prime).any()
        # h_prime: N x out
        h_prime = h_prime.div(e_rowsum)
        new_h_prime = new_h_prime.div(new_e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()
        assert not torch.isnan(new_h_prime).any()

        
        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime), F.elu(new_h_prime)
        else:
            # if this layer is last layer,
            return h_prime, new_h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class SelfAttention(nn.Module):
    """
    Multi-head attention as per https://arxiv.org/pdf/1706.03762.pdf
    Refer Figure 2
    """
    def __init__(self, input_depth, total_key_depth, total_value_depth, output_depth, 
                 num_heads, seq_length, bias_mask=None, dropout=0.0): #
        """
        Parameters:
            input_depth: Size of last dimension of input 100
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head 100
            total_value_depth: Size of last dimension of values. Must be divisible by num_head 100
            output_depth: Size last dimension of the final output 200
            seq_length: 4
            num_heads: Number of attention heads 4
            bias_mask: Masking tensor to prevent connections to future elements
            dropout: Dropout probability (Should be non-zero only during training)
        """
        super(SelfAttention, self).__init__()
        if total_key_depth % num_heads != 0:
            raise ValueError("Key depth (%d) must be divisible by the number of "
                             "attention heads (%d)." % (total_key_depth, num_heads))
        if total_value_depth % num_heads != 0:
            raise ValueError("Value depth (%d) must be divisible by the number of "
                             "attention heads (%d)." % (total_value_depth, num_heads))
            
        self.num_heads = num_heads
        self.query_scale = (total_key_depth//num_heads)**-0.5
        self.bias_mask = bias_mask
        
        # Key and query depth will be same
        self.query_linear = nn.Linear(input_depth, total_key_depth, bias=False)
        self.key_linear = nn.Linear(input_depth, total_key_depth, bias=False)
        self.value_linear = nn.Linear(input_depth, total_value_depth, bias=False)
        self.output_linear = nn.Linear(total_value_depth, input_depth, bias=False)
        self.layer_norm = nn.LayerNorm(input_depth)
        self.final_linear = nn.Linear(input_depth * seq_length, output_depth, bias=False)
		
        self.dropout = nn.Dropout(dropout)
    
    def _split_heads(self, x):
        """
        Split x such to add an extra num_heads dimension
        Input:
            x: a Tensor with shape [batch_size, seq_length, depth]
        Returns:
            A Tensor with shape [batch_size, num_heads, seq_length, depth/num_heads]
        """
        if len(x.shape) != 3:
            raise ValueError("x must have rank 3")
        shape = x.shape
        return x.view(shape[0], shape[1], self.num_heads, shape[2]//self.num_heads).permute(0, 2, 1, 3)
    
    def _merge_heads(self, x):
        """
        Merge the extra num_heads into the last dimension
        Input:
            x: a Tensor with shape [batch_size, num_heads, seq_length, depth/num_heads]
        Returns:
            A Tensor with shape [batch_size, seq_length, depth]
        """
        if len(x.shape) != 4:
            raise ValueError("x must have rank 4")
        shape = x.shape
        return x.permute(0, 2, 1, 3).contiguous().view(shape[0], shape[2], shape[3]*self.num_heads)
       
    def _merge_to_seq(self, x):
        if len(x.shape) != 3:
            raise ValueError("x must have rank 4")
        shape = x.shape
        return x.view(shape[0], shape[1] * shape[2])
	
    def forward(self, queries, keys, values):
        
        # Do a linear for each component
        residual = queries
        queries = self.query_linear(queries)
        keys = self.key_linear(keys)
        values = self.value_linear(values)
        
        # Split into multiple heads
        queries = self._split_heads(queries)
        keys = self._split_heads(keys)
        values = self._split_heads(values)
        
        # Scale queries
        queries *= self.query_scale
        
        # Combine queries and keys
        logits = torch.matmul(queries, keys.permute(0, 1, 3, 2))
        
        # Add bias to mask future values
        if self.bias_mask is not None:
            logits += self.bias_mask[:, :, :logits.shape[-2], :logits.shape[-1]].type_as(logits.data)
        
        # Convert to probabilites
        weights = nn.functional.softmax(logits, dim=-1)
        
        # Dropout
        weights = self.dropout(weights)
        
        # Combine with values to get context
        contexts = torch.matmul(weights, values)
        
        # Merge heads
        contexts = self._merge_heads(contexts)
        #contexts = torch.tanh(contexts)
        
        # Linear to get output
        outputs = self.output_linear(contexts)
        outputs = self.dropout(outputs)
        outputs = self.layer_norm(residual + outputs)
        outputs = self.final_linear(self._merge_to_seq(outputs))
		
        return outputs
