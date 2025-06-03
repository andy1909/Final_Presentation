from VRP_1_Input import VRPInputConfig

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
from torch_geometric.data import Data
from torch_geometric.nn import TransformerConv

np.random.seed(42)

# =====================================================
# --- 1. Định nghĩa mô hình GNN (TransformerConv) ---
# =====================================================

class GNNEdgeAttr(nn.Module):
    def __init__(self, num_node_features, gnn_hidden_dim, embedding_dim, edge_feature_dim, num_heads):
        super(GNNEdgeAttr, self).__init__()
        self.conv1 = TransformerConv(num_node_features, gnn_hidden_dim, heads=num_heads, dropout=0.3, edge_dim=edge_feature_dim)
        self.conv2 = TransformerConv(gnn_hidden_dim * num_heads, gnn_hidden_dim, heads=num_heads, dropout=0.3, edge_dim=edge_feature_dim)
        self.conv3 = TransformerConv(gnn_hidden_dim * num_heads, embedding_dim, heads=1, concat=False, dropout=0.3, edge_dim=edge_feature_dim)
        self.dropout = nn.Dropout(0.3) # Dropout tránh overfitting

    def forward(self, data_GNN: Data):
        x, edge_index, edge_attr = data_GNN.x, data_GNN.edge_index, data_GNN.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x = self.conv3(x, edge_index, edge_attr)
        return x

# =====================================================
# --- 2. Tạo features cho GNN ---
# =====================================================

class GNNFeature:
    def __init__(self, config: VRPInputConfig):
        self.config                     = config
        self.num_nodes                  = config.num_nodes
        self.penalty_per_unvisited_node = config.penalty_per_unvisited_node
        self.demand                     = config.demand
        self.service_time               = config.service_time
        self.coords                     = config.coords
        self.distance                   = config.distance
        self.vehicle_types              = config.vehicle_types
        self.noise_avg_velocity_types   = config.noise_avg_velocity_types

    def build(self):
        original_node_features_np = np.array(
            [[p, d, s, c[0], c[1]] for p, d, s, c in zip(self.penalty_per_unvisited_node, self.demand, self.service_time, self.coords)],
            dtype=np.float32
        )
        edge_index_list = []
        edge_attr_list = []
        num_vehicle_types = len(self.vehicle_types)
        for i, j in itertools.permutations(range(self.num_nodes), 2):
            edge_index_list.append([i, j])
            distance = self.distance[i][j]
            current_edge_features = [distance]
            for type_idx in range(num_vehicle_types):
                vehicle_config = self.vehicle_types[type_idx]
                noise_avg_velocity_type = self.noise_avg_velocity_types[type_idx]
                fuel_cost_per_100km = vehicle_config['fuel_cost_100km']
                avg_velocity_ij = noise_avg_velocity_type[i][j]
                travel_time_minutes = distance / (avg_velocity_ij / 60) if avg_velocity_ij > 0 else float('inf')
                fuel_cost_edge = distance * (fuel_cost_per_100km / 100)
                current_edge_features.extend([travel_time_minutes, fuel_cost_edge])
            edge_attr_list.append(current_edge_features)
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr_list, dtype=torch.float32)
        node_features = torch.tensor(original_node_features_np, dtype=torch.float32)
        self.edge_feature_dim = edge_attr.shape[1]
        data_GNN = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)

        return data_GNN

# =====================================================
# --- 3. Định nghĩa Decoder ---
# =====================================================

class AttentionDecoder(nn.Module):
    def __init__(self, embedding_dim, decoder_hidden_dim):
        super(AttentionDecoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.W_query = nn.Linear(embedding_dim + embedding_dim, decoder_hidden_dim, bias=False)
        self.W_key = nn.Linear(embedding_dim, decoder_hidden_dim, bias=False)

    def forward(self, current_node_emb, context_emb, candidate_node_embs, mask):
        combined_input = torch.cat([current_node_emb, context_emb], dim=1)
        query = self.W_query(combined_input)
        keys = self.W_key(candidate_node_embs)
        attn_scores = torch.matmul(query, keys.t())

        # --- 4.1 Tạo mask ---
        masked_attn_scores = attn_scores.masked_fill(mask == 0, -float('inf'))

        # --- 4.2 Phân phối xác suất và chọn hành động ---
        probs = F.softmax(masked_attn_scores, dim=1)
        k = min(50, max(1, int(0.5 * probs.shape[1])))
        top_probs, _ = torch.topk(probs, k, dim=1)
        threshold = top_probs[:, -1].unsqueeze(1)
        top_mask = (probs >= threshold)
        filtered_probs = probs * top_mask
        filtered_probs = filtered_probs / (filtered_probs.sum(dim=1, keepdim=True) + 1e-10)
        dist = torch.distributions.Categorical(filtered_probs)
        action_idx = dist.sample()
        log_prob_action = dist.log_prob(action_idx)
        log_probs_all = torch.log(filtered_probs + 1e-10)

        return log_probs_all, log_prob_action, action_idx.item()

# =====================================================
# --- 4. Context Embedding ---
# =====================================================

class ContextEmbedding(nn.Module):
    def __init__(self, embedding_dim, vehicle_feature_dim):
        super(ContextEmbedding, self).__init__()
        self.embedding_dim = embedding_dim
        self.vehicle_feature_dim = vehicle_feature_dim
        self.vehicle_state_processor = nn.Sequential(
            nn.Linear(vehicle_feature_dim, embedding_dim),
            nn.ReLU()
        )
        combined_input_dim = embedding_dim * 4
        self.final_combiner = nn.Sequential(
            nn.Linear(combined_input_dim, embedding_dim),
            nn.ReLU()
        )

    def forward(self,
                current_load: float,
                current_vehicle_capacity: float,
                current_time: float,
                current_vehicle_max_time: float,
                all_node_embeddings: torch.Tensor,
                visited_mask: torch.Tensor,
                current_node_idx: int,
                depot_idx: int
               ):

        # --- 6.1. Trạng thái của xe ---
        device = all_node_embeddings.device
        remaining_capacity = max(0.0, current_vehicle_capacity - current_load)
        remaining_time = max(0.0, current_vehicle_max_time - current_time)
        epsilon = 1e-6
        vehicle_features_np = [
            current_load / (current_vehicle_capacity + epsilon),                                                # Tải trọng hiện tại / sức chứa Max
            remaining_capacity / (current_vehicle_capacity + epsilon),                                          # Tải trọng còn lại / sức chứa Max
            remaining_time / (current_vehicle_max_time + epsilon)                                               # Time còn lại / Time tối đa
        ]
        vehicle_features = torch.as_tensor(
            [vehicle_features_np], dtype=torch.float32, device=device
        )
        vehicle_state_emb = self.vehicle_state_processor(vehicle_features)

        # --- 6.2. Trạng thái toàn cục ---
        depot_emb = all_node_embeddings[depot_idx].unsqueeze(0)                                                 # Emb của Kho
        current_node_emb = all_node_embeddings[current_node_idx].unsqueeze(0)                                   # Emb của Node hiện tại
        unvisited_node_indices = (~visited_mask).nonzero(as_tuple=True)[0]                                      # Emb avg của Node chưa thăm
        if unvisited_node_indices.numel() > 0:
            unvisited_embs = all_node_embeddings[unvisited_node_indices]
            mean_unvisited_emb = unvisited_embs.mean(dim=0, keepdim=True)
        else:
            mean_unvisited_emb = torch.zeros((1, self.embedding_dim), device=device)

        combined_embedding = torch.cat(                                                                         # Combine Emb
            [vehicle_state_emb, mean_unvisited_emb, current_node_emb, depot_emb], dim=1
        )
        dynamic_context_embedding = self.final_combiner(combined_embedding)

        return dynamic_context_embedding

# =====================================================
# --- 5. Critic ---
# =====================================================

class CriticValueNetwork(nn.Module):
    def __init__(self, embedding_dim, critic_hidden_dim):
        super().__init__()
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, critic_hidden_dim),
            nn.BatchNorm1d(critic_hidden_dim, affine=True),
            nn.ReLU(),
            nn.Linear(critic_hidden_dim, critic_hidden_dim // 2),
            nn.BatchNorm1d(critic_hidden_dim // 2, affine=True),
            nn.ReLU(),
            nn.Linear(critic_hidden_dim // 2, 1)
        )

    def forward(self, dynamic_context_embedding):
        normalized_dc_emb = self.layer_norm(dynamic_context_embedding)
        return self.mlp(normalized_dc_emb)

# =====================================================
# --- 6. Khởi tạo GNN, Decoder và Critic ---
# =====================================================

class ModelComponents:
    def __init__(self, data_GNN: Data, feature_GNN: GNNFeature,
                gnn_hidden_dim: int, embedding_dim: int, num_heads: int,
                decoder_hidden_dim: int, critic_hidden_dim: int,
                vehicle_feature_dim: int = 3
                ):
        self.config_params = {
            'gnn_hidden_dim': gnn_hidden_dim,
            'embedding_dim': embedding_dim,
            'num_heads': num_heads,
            'decoder_hidden_dim': decoder_hidden_dim,
            'critic_hidden_dim': critic_hidden_dim,
            'vehicle_feature_dim': vehicle_feature_dim
        }

        num_node_features = data_GNN.x.shape[1]
        edge_feature_dim = feature_GNN.edge_feature_dim

        self.gnn_encoder = GNNEdgeAttr(num_node_features, gnn_hidden_dim, embedding_dim, edge_feature_dim, num_heads)