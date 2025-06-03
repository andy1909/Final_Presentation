from VRP_2_Model import AttentionDecoder, CriticValueNetwork

import torch
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.distributions import CategoricalDistribution, Distribution
from typing import Callable, Dict, Optional, Tuple

# =====================================================
# --- 9. Policy tùy chỉnh cho PPO ---
# =====================================================

class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, embedding_dim: int):
        super().__init__(observation_space, features_dim=1)
        self.embedding_dim = embedding_dim

    def forward(self, observations: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        processed_obs = {}
        for key, tensor in observations.items():
            if key != 'mask' and tensor.dtype != torch.float32:
                processed_obs[key] = tensor.to(torch.float32)
            else:
                processed_obs[key] = tensor
        return processed_obs

# =====================================================
# --- 10. Policy Actor Critic ---
# =====================================================

class VRPActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        node_embeddings_tensor = kwargs.pop("node_embeddings")
        self.node_embeddings_cpu = node_embeddings_tensor.cpu()
        self.embedding_dim = kwargs.pop("embedding_dim")
        self.decoder_hidden_dim = kwargs.pop("decoder_hidden_dim")
        self.critic_hidden_dim = kwargs.pop("critic_hidden_dim")

        if "features_extractor_class" not in kwargs:
            kwargs["features_extractor_class"] = CustomFeatureExtractor
        if "features_extractor_kwargs" not in kwargs:
            kwargs["features_extractor_kwargs"] = {}
        kwargs["features_extractor_kwargs"]["embedding_dim"] = self.embedding_dim

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs,
        )
        self.ortho_init = False

    def _build(self, lr_schedule: Callable[[float], float]) -> None:
        # --- 10.1.1. Xử lý Emb các node ---
        self.node_embeddings_device = self.node_embeddings_cpu.to(self.device)
        current_embedding_dim = self.embedding_dim
        current_decoder_hidden_dim = self.decoder_hidden_dim
        current_critic_hidden_dim = self.critic_hidden_dim

        # --- 10.1.2. Định nghĩa Action Distribution ---
        self.action_dist = CategoricalDistribution(self.action_space.n)

        # --- 10.1.3. Khởi tạo mạng Actor (AttentionDecoder) ---
        self.action_net = AttentionDecoder(
            embedding_dim=current_embedding_dim,
            decoder_hidden_dim=current_decoder_hidden_dim
        ).to(self.device)

        # --- 10.1.4. Khởi tạo mạng Critic (CriticValueNetwork) ---
        self.value_net = CriticValueNetwork(
            embedding_dim=current_embedding_dim,
            critic_hidden_dim=current_critic_hidden_dim
        ).to(self.device)

        # --- 10.1.5. Thiết lập Optimizer ---
        policy_params = list(self.action_net.parameters()) + list(self.value_net.parameters())
        if self.features_extractor is not None and hasattr(self.features_extractor, 'parameters'):
            try:
                extractor_params = list(self.features_extractor.parameters())
                if extractor_params:
                    policy_params += extractor_params
            except StopIteration:
                pass
        self.optimizer = self.optimizer_class(policy_params, lr=lr_schedule(1), **self.optimizer_kwargs)

    def get_distribution(self, observation: Dict[str, torch.Tensor]) -> Distribution:
        # Trích xuất đặc trưng từ observation (sử dụng CustomFeatureExtractor)
        features = self.extract_features(observation) # features là một dict
        context_embedding = features["context"]
        current_node_emb = features["current_node_emb"]
        mask = features["mask"].bool() # Chuyển mask sang kiểu boolean
        all_node_embs_device = self.node_embeddings_device

        # Tính action_logits (đây chính là "latent_pi" trong ngữ cảnh của chúng ta)
        batch_size = context_embedding.shape[0]
        if all_node_embs_device.dim() == 2: # (num_nodes, embedding_dim)
            all_node_embeddings_batch = all_node_embs_device.unsqueeze(0).expand(batch_size, -1, -1) # (batch_size, num_nodes, embedding_dim)
        else: # Đã có dạng batch rồi
            all_node_embeddings_batch = all_node_embs_device
        
        combined_input = torch.cat([current_node_emb, context_embedding], dim=1)
        query = self.action_net.W_query(combined_input).unsqueeze(1) # (batch_size, 1, decoder_hidden_dim)
        keys = self.action_net.W_key(all_node_embeddings_batch) # (batch_size, num_nodes, decoder_hidden_dim)
        
        attn_scores = torch.bmm(query, keys.transpose(1, 2)).squeeze(1) # (batch_size, num_nodes)
        masked_attn_scores = attn_scores.masked_fill(~mask, -float('inf'))
        action_logits = masked_attn_scores # Đây là latent_pi

        # self.action_dist là CategoricalDistribution đã khởi tạo trong _build
        return self.action_dist.proba_distribution(action_logits=action_logits)

    # --- 10.2. Hàm quyết định hành động tiếp theo ---
    def forward(self, obs: Dict[str, torch.Tensor], deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.extract_features(obs)
        context_embedding = features["context"]
        current_node_emb = features["current_node_emb"]
        mask = features["mask"].bool()
        all_node_embs_device = self.node_embeddings_device

        # --- 10.2.1. Tính Vaule Critic ---
        values = self.value_net(context_embedding)

        # --- 10.2.2. Tính Logits Actor (Attention) ---
        batch_size = context_embedding.shape[0]
        if all_node_embs_device.dim() == 2:
            all_node_embeddings_batch = all_node_embs_device.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            all_node_embeddings_batch = all_node_embs_device
        combined_input = torch.cat([current_node_emb, context_embedding], dim=1)
        query = self.action_net.W_query(combined_input).unsqueeze(1)
        keys = self.action_net.W_key(all_node_embeddings_batch)

        attn_scores = torch.bmm(query, keys.transpose(1, 2)).squeeze(1)
        masked_attn_scores = attn_scores.masked_fill(~mask, -float('inf'))
        action_logits = masked_attn_scores

        distribution = self.action_dist.proba_distribution(action_logits=action_logits)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)

        return actions, values, log_prob

    # --- 10.3. Hàm tính toán hành động tiếp theo ---
    def evaluate_actions(self, obs: Dict[str, torch.Tensor], actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        features = self.extract_features(obs)
        context_embedding = features["context"]
        current_node_emb = features["current_node_emb"]
        mask = features["mask"].bool()
        all_node_embs_device = self.node_embeddings_device

        # --- 10.3.1. Critic ---
        values = self.value_net(context_embedding)

        # --- 10.3.2. Actor (Attention) ---
        batch_size = context_embedding.shape[0]
        if all_node_embs_device.dim() == 2:
            all_node_embeddings_batch = all_node_embs_device.unsqueeze(0).expand(batch_size, -1, -1)
        else:
             all_node_embeddings_batch = all_node_embs_device

        combined_input = torch.cat([current_node_emb, context_embedding], dim=1)
        query = self.action_net.W_query(combined_input).unsqueeze(1)
        keys = self.action_net.W_key(all_node_embeddings_batch)

        attn_scores = torch.bmm(query, keys.transpose(1, 2)).squeeze(1)
        masked_attn_scores = attn_scores.masked_fill(~mask, -float('inf'))
        action_logits = masked_attn_scores

        distribution = self.action_dist.proba_distribution(action_logits=action_logits)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()

        return values, log_prob, entropy

    # --- 10.4. Hàm tính toán giá trị hành động tiếp theo ---
    def predict_values(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        features = self.extract_features(obs)
        context_embedding = features["context"]
        values = self.value_net(context_embedding)
        return values