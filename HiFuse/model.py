import torch
import torch.nn as nn
import torch.nn.functional as F


class HiFuseNetwork(nn.Module):
    def __init__(
        self,
        dim_in_feat_omics1,
        dim_out_feat_omics1,
        dim_in_feat_omics2,
        dim_out_feat_omics2,
        dropout=0.0,
        act=F.relu,
        fusion_hidden_dim=256,
    ):
        super().__init__()

        self.encoder1 = GraphPropagation(dim_in_feat_omics1, dim_out_feat_omics1)
        self.decoder1 = GraphPropagation(dim_out_feat_omics1, dim_in_feat_omics1)
        self.encoder2 = GraphPropagation(dim_in_feat_omics2, dim_out_feat_omics2)
        self.decoder2 = GraphPropagation(dim_out_feat_omics2, dim_in_feat_omics2)

        self.graph_view_attention1 = PairwiseAttention(
            dim_out_feat_omics1,
            dim_out_feat_omics1,
        )
        self.graph_view_attention2 = PairwiseAttention(
            dim_out_feat_omics2,
            dim_out_feat_omics2,
        )
        self.modality_attention = PairwiseAttention(
            dim_out_feat_omics1,
            dim_out_feat_omics2,
        )
        self.consensus1 = ConsensusProjection(
            dim_out_feat_omics1,
            dim_out_feat_omics1,
        )
        self.consensus2 = ConsensusProjection(
            dim_out_feat_omics2,
            dim_out_feat_omics2,
        )
        self.complementary_fusion = ComplementaryFusion(
            dim_out_feat_omics1 + dim_out_feat_omics2,
            fusion_hidden_dim,
            dim_out_feat_omics1,
        )
        self.path_attention = PairwiseAttention(
            dim_out_feat_omics1,
            dim_out_feat_omics1,
        )

    def forward(
        self,
        features_omics1,
        features_omics2,
        adj_spatial_omics1,
        adj_feature_omics1,
        adj_spatial_omics2,
        adj_feature_omics2,
    ):
        spatial1 = self.encoder1(features_omics1, adj_spatial_omics1)
        spatial2 = self.encoder2(features_omics2, adj_spatial_omics2)
        feature1 = self.encoder1(features_omics1, adj_feature_omics1)
        feature2 = self.encoder2(features_omics2, adj_feature_omics2)

        modality1, graph_weights1 = self.graph_view_attention1(spatial1, feature1)
        modality2, graph_weights2 = self.graph_view_attention2(spatial2, feature2)
        attention_fusion, modality_weights = self.modality_attention(
            modality1,
            modality2,
        )
        consensus1 = self.consensus1(modality1)
        consensus2 = self.consensus2(modality2)
        complementary_fusion = self.complementary_fusion(modality1, modality2)
        joint, path_weights = self.path_attention(
            attention_fusion,
            complementary_fusion,
        )

        reconstruction1 = self.decoder1(joint, adj_spatial_omics1)
        reconstruction2 = self.decoder2(joint, adj_spatial_omics2)
        cycle1 = self.encoder2(
            self.decoder2(modality1, adj_spatial_omics2),
            adj_spatial_omics2,
        )
        cycle2 = self.encoder1(
            self.decoder1(modality2, adj_spatial_omics1),
            adj_spatial_omics1,
        )

        return {
            "modality1": modality1,
            "modality2": modality2,
            "joint": joint,
            "attention_fusion": attention_fusion,
            "complementary_fusion": complementary_fusion,
            "consensus1": consensus1,
            "consensus2": consensus2,
            "reconstruction1": reconstruction1,
            "reconstruction2": reconstruction2,
            "cycle1": cycle1,
            "cycle2": cycle2,
            "graph_weights1": graph_weights1,
            "graph_weights2": graph_weights2,
            "modality_weights": modality_weights,
            "path_weights": path_weights,
        }

class GraphPropagation(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, features, adjacency):
        projected = torch.mm(features, self.weight)
        return torch.spmm(adjacency, projected)


class PairwiseAttention(nn.Module):
    def __init__(self, in_features, attention_features):
        super().__init__()
        self.projection = nn.Parameter(
            torch.FloatTensor(in_features, attention_features)
        )
        self.context = nn.Parameter(torch.FloatTensor(attention_features, 1))
        nn.init.xavier_uniform_(self.projection)
        nn.init.xavier_uniform_(self.context)

    def forward(self, first, second):
        stacked = torch.cat(
            [
                torch.unsqueeze(torch.squeeze(first), dim=1),
                torch.unsqueeze(torch.squeeze(second), dim=1),
            ],
            dim=1,
        )
        scores = torch.matmul(
            F.tanh(torch.matmul(stacked, self.projection)), self.context
        )
        weights = F.softmax(torch.squeeze(scores, dim=-1) + 1e-6, dim=1)
        combined = torch.matmul(
            torch.transpose(stacked, 1, 2),
            torch.unsqueeze(weights, -1),
        )
        return torch.squeeze(combined), weights


class ComplementaryFusion(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, first, second):
        return self.network(torch.cat([first, second], dim=1))


class ConsensusProjection(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(input_dim, output_dim))

    def forward(self, embedding):
        return self.network(embedding)
