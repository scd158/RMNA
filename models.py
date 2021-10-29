import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import time
from layers import SpGraphAttentionLayer, ConvKB, SelfAttention

CUDA = torch.cuda.is_available()  # checking cuda availability


class SpGAT(nn.Module):
    def __init__(self, num_nodes, nfeat, nhid, sa_key_dim, sa_value_dim, relation_dim, dropout, alpha, nheads,
                 nsaheads):
        """
            Sparse version of GAT
            nfeat -> Entity Input Embedding dimensions 50
            nhid  -> Entity Output Embedding dimensions 100
            sa_key_dim, sa_value_dim -> key and value dimentions of self-attention 100
            relation_dim -> Relation Embedding dimensions 50
            num_nodes -> number of nodes in the Graph
            nheads -> Used for Multihead attention 2
            nsaheads -> Used for self attention 4

        """
        super(SpGAT, self).__init__()
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(self.dropout)
        self.attentions = [SpGraphAttentionLayer(num_nodes, nfeat,
                                                 nhid,
                                                 relation_dim,
                                                 dropout=dropout,
                                                 alpha=alpha,
                                                 concat=True)
                           for _ in range(nheads)]
        self.self_attention1 = SelfAttention(nhid, nhid, nhid, nhid * nheads, nsaheads, 2 * nheads)
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        # W matrix to convert h_input to h_output dimension
        self.W = nn.Parameter(torch.zeros(size=(relation_dim, nheads * nhid)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.out_att = SpGraphAttentionLayer(num_nodes, nhid * nheads,
                                             nheads * nhid, nheads * nhid,
                                             dropout=dropout,
                                             alpha=alpha,
                                             concat=False
                                             )
        self.self_attention2 = SelfAttention(nhid * nheads, nhid * nheads, nhid * nheads, nhid * nheads, nsaheads, 2)

    def forward(self, Corpus_, batch_inputs, entity_embeddings, relation_embed, edge_list,
                new_edge_list, edge_type, new_edge_type, edge_embed, new_edge_embed, new_edge_other, edge_list_nhop,
                edge_type_nhop):
        x = entity_embeddings
        if (edge_type_nhop.size()[0] != 0):
            edge_embed_nhop = relation_embed[edge_type_nhop[:, 0]] + relation_embed[edge_type_nhop[:, 1]]
        else:
            edge_embed_nhop = edge_type_nhop
        att_res = [att(x, edge_list, new_edge_list, edge_embed, new_edge_embed, new_edge_other, edge_list_nhop,
                       edge_embed_nhop) for att in self.attentions]
        head_1 = torch.cat([x[0].unsqueeze(0) for x in att_res], dim=0)
        new_head_1 = torch.cat([x[1].unsqueeze(0) for x in att_res], dim=0)
        head_1 = head_1.permute(1, 0, 2)
        new_head_1 = new_head_1.permute(1, 0, 2)  # N * nheads * dim
        x = torch.cat((head_1, new_head_1), dim=1)
        x = self.self_attention1(x, x, x)

        x = self.dropout_layer(x)  # (N*200)
        out_relation_1 = relation_embed.mm(self.W)

        edge_embed = out_relation_1[edge_type]
        new_edge_embed = out_relation_1[new_edge_type]
        if (edge_type_nhop.size()[0] != 0):
            edge_embed_nhop = out_relation_1[
                                  edge_type_nhop[:, 0]] + out_relation_1[edge_type_nhop[:, 1]]
        else:
            edge_embed_nhop = edge_type_nhop

        head_2, new_head_2 = self.out_att(x, edge_list, new_edge_list, edge_embed, new_edge_embed, new_edge_other,
                                          edge_list_nhop, edge_embed_nhop)  # N * dim
        head_2 = torch.unsqueeze(F.elu(head_2), 1)
        new_head_2 = torch.unsqueeze(F.elu(new_head_2), 1)
        x = torch.cat((head_2, new_head_2), dim=1)
        x = self.self_attention2(x, x, x)
        return x, out_relation_1


class SpKBGATModified(nn.Module):
    def __init__(self, initial_entity_emb, initial_relation_emb, entity_out_dim, sa_dim, relation_out_dim,
                 drop_GAT, alpha, nheads_GAT, nheads_sa):
        '''Sparse version of KBGAT
        entity_in_dim -> Entity Input Embedding dimensions 50
        entity_out_dim  -> Entity Output Embedding dimensions, passed as a list [100,200]
	    sa_dim -> [200,200]
        num_relation -> number of unique relations
        relation_dim -> Relation Embedding dimensions [100,200]
        num_nodes -> number of nodes in the Graph
        nheads_GAT -> Used for Multihead attention, passed as a list [2,2]'''

        super().__init__()

        self.num_nodes = initial_entity_emb.shape[0]  # N
        self.entity_in_dim = initial_entity_emb.shape[1]  # 50
        self.entity_out_dim_1 = entity_out_dim[0]  # 100
        self.nheads_GAT_1 = nheads_GAT[0]  # 2
        self.entity_out_dim_2 = entity_out_dim[1]  # 200
        self.nheads_GAT_2 = nheads_GAT[1]  # 2
        self.nheads_sa_1 = nheads_sa[0]
        self.nheads_sa_2 = nheads_sa[1]
        self.sa_key_dim = sa_dim[0]  # 100
        self.sa_value_dim = sa_dim[1]  # 100
        # Properties of Relations
        self.num_relation = initial_relation_emb.shape[0]  # NR
        self.relation_dim = initial_relation_emb.shape[1]  # 50
        self.relation_out_dim_1 = relation_out_dim[0]  # 100

        self.drop_GAT = drop_GAT
        self.alpha = alpha  # For leaky relu

        self.final_entity_embeddings = nn.Parameter(
            torch.randn(self.num_nodes, self.entity_out_dim_1 * self.nheads_GAT_1))  # N * 200

        self.final_relation_embeddings = nn.Parameter(
            torch.randn(self.num_relation, self.entity_out_dim_1 * self.nheads_GAT_1))

        self.entity_embeddings = nn.Parameter(initial_entity_emb)
        self.relation_embeddings = nn.Parameter(initial_relation_emb)

        self.sparse_gat_1 = SpGAT(self.num_nodes, self.entity_in_dim, self.entity_out_dim_1, self.sa_key_dim,
                                  self.sa_value_dim, self.relation_dim,
                                  self.drop_GAT, self.alpha, self.nheads_GAT_1, self.nheads_sa_1)

        self.W_entities = nn.Parameter(torch.zeros(
            size=(self.entity_in_dim, self.entity_out_dim_1 * self.nheads_GAT_1)))
        nn.init.xavier_uniform_(self.W_entities.data, gain=1.414)

    def forward(self, Corpus_, adj, new_adj, batch_inputs,
                train_indices_nhop):  # newadj:row, col, head_cov. conf, rule_length
        # getting edge list
        edge_list = adj[0]  # [[tails],[heads]]
        edge_type = adj[1]  # [relations]

        new_edge_list = new_adj[0]  # [[tails],[heads]]
        new_edge_type = new_adj[1]  # [relations]
        new_edge_other = new_adj[2]  # hc, conf, rl, score

        if (train_indices_nhop.size()[0] != 0):
            edge_list_nhop = torch.cat(
                (train_indices_nhop[:, 3].unsqueeze(-1), train_indices_nhop[:, 0].unsqueeze(-1)), dim=1).t()
            edge_type_nhop = torch.cat(
                [train_indices_nhop[:, 1].unsqueeze(-1), train_indices_nhop[:, 2].unsqueeze(-1)], dim=1)
        else:
            edge_list_nhop = train_indices_nhop
            edge_type_nhop = train_indices_nhop

        if (CUDA):
            edge_list = edge_list.cuda()
            edge_type = edge_type.cuda()
            edge_list_nhop = edge_list_nhop.cuda()
            edge_type_nhop = edge_type_nhop.cuda()
            new_edge_list = new_edge_list.cuda()
            new_edge_type = new_edge_type.cuda()
            new_edge_other = new_edge_other.cuda()
        edge_embed = self.relation_embeddings[edge_type]
        new_edge_embed = self.relation_embeddings[new_edge_type]

        start = time.time()

        self.entity_embeddings.data = F.normalize(
            self.entity_embeddings.data, p=2, dim=1).detach()
        new_edge_other = F.normalize(new_edge_other, p=2, dim=1)

        # self.relation_embeddings.data = F.normalize(
        #     self.relation_embeddings.data, p=2, dim=1)

        out_entity_1, out_relation_1 = self.sparse_gat_1(
            Corpus_, batch_inputs, self.entity_embeddings, self.relation_embeddings,
            edge_list, new_edge_list, edge_type, new_edge_type, edge_embed, new_edge_embed, new_edge_other,
            edge_list_nhop, edge_type_nhop)

        mask_indices = torch.unique(batch_inputs[:, 2]).cuda()
        mask = torch.zeros(self.entity_embeddings.shape[0]).cuda()
        mask[mask_indices] = 1.0

        entities_upgraded = self.entity_embeddings.mm(self.W_entities)
        out_entity_1 = entities_upgraded + \
                       mask.unsqueeze(-1).expand_as(out_entity_1) * out_entity_1

        out_entity_1 = F.normalize(out_entity_1, p=2, dim=1)

        self.final_entity_embeddings.data = out_entity_1.data
        self.final_relation_embeddings.data = out_relation_1.data

        return out_entity_1, out_relation_1


class SpKBGATConvOnly(nn.Module):
    def __init__(self, initial_entity_emb, initial_relation_emb, entity_out_dim, relation_out_dim,
                 drop_GAT, drop_conv, alpha, alpha_conv, nheads_GAT, conv_out_channels):
        '''Sparse version of KBGAT
        entity_in_dim -> Entity Input Embedding dimensions
        entity_out_dim  -> Entity Output Embedding dimensions, passed as a list
        num_relation -> number of unique relations
        relation_dim -> Relation Embedding dimensions
        num_nodes -> number of nodes in the Graph
        nheads_GAT -> Used for Multihead attention, passed as a list '''

        super().__init__()

        self.num_nodes = initial_entity_emb.shape[0]
        self.entity_in_dim = initial_entity_emb.shape[1]
        self.entity_out_dim_1 = entity_out_dim[0]
        self.nheads_GAT_1 = nheads_GAT[0]
        self.entity_out_dim_2 = entity_out_dim[1]
        self.nheads_GAT_2 = nheads_GAT[1]

        # Properties of Relations
        self.num_relation = initial_relation_emb.shape[0]
        self.relation_dim = initial_relation_emb.shape[1]
        self.relation_out_dim_1 = relation_out_dim[0]

        self.drop_GAT = drop_GAT
        self.drop_conv = drop_conv
        self.alpha = alpha  # For leaky relu
        self.alpha_conv = alpha_conv
        self.conv_out_channels = conv_out_channels

        self.final_entity_embeddings = nn.Parameter(
            torch.randn(self.num_nodes, self.entity_out_dim_1 * self.nheads_GAT_1))

        self.final_relation_embeddings = nn.Parameter(
            torch.randn(self.num_relation, self.entity_out_dim_1 * self.nheads_GAT_1))

        self.convKB = ConvKB(self.entity_out_dim_1 * self.nheads_GAT_1, 3, 1,
                             self.conv_out_channels, self.drop_conv, self.alpha_conv)

    def forward(self, Corpus_, adj, batch_inputs):
        conv_input = torch.cat(
            (self.final_entity_embeddings[batch_inputs[:, 0], :].unsqueeze(1), self.final_relation_embeddings[
                batch_inputs[:, 1]].unsqueeze(1), self.final_entity_embeddings[batch_inputs[:, 2], :].unsqueeze(1)),
            dim=1)
        out_conv = self.convKB(conv_input)
        return out_conv

    def batch_test(self, batch_inputs):
        conv_input = torch.cat(
            (self.final_entity_embeddings[batch_inputs[:, 0], :].unsqueeze(1), self.final_relation_embeddings[
                batch_inputs[:, 1]].unsqueeze(1), self.final_entity_embeddings[batch_inputs[:, 2], :].unsqueeze(1)),
            dim=1)
        out_conv = self.convKB(conv_input)
        return out_conv
