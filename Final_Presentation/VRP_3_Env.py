from VRP_1_Input import VRPInputConfig
from VRP_2_Model import ContextEmbedding, GNNFeature

import numpy as np
import torch
import copy
import gymnasium as gym
from gymnasium import spaces
from typing import Callable

class VRPEnvGym(gym.Env):
    metadata = {"render_modes": [], "render_fps": 4}

    # --- 1.1. Hàm khởi tạo môi trường ---
    def __init__(self,
                config_params: dict,
                node_embeddings: torch.Tensor,
                context_embedder: ContextEmbedding,
                ):
        super().__init__()

        # 1.1.1. Khởi tạo map 
        current_config_params = copy.deepcopy(config_params)
        self.config = VRPInputConfig(**current_config_params)
        self.num_nodes                  = self.config.num_nodes                                                         # Số cửa hàng và kho
        self.depot_idx                  = self.config.depot_idx                                                         # ID kho (0)
        self.coords                     = self.config.coords                                                            # Toạ độ (Kinh độ, Vĩ độ)
        self.distance                   = self.config.distance                                                          # Ma trận khoảng cách (km)
        self.demand                     = self.config.demand                                                            # Nhu cầu từng cửa hàng (kg)
        self.service_time               = self.config.service_time                                                      # Thời gian phục vụ từng cửa hàng (Phút)
        self.penalty_per_unvisited_node = self.config.penalty_per_unvisited_node                                        # Tiền phạt khi không thăm cửa hàng (VNĐ)

        # 1.1.2. Khởi tạo xe
        self.vehicle_types              = self.config.vehicle_types                                                     # Số loại xe
        self.noise_avg_velocity_types   = self.config.noise_avg_velocity_types                                          # Vận tốc đã bị nhiễu
        self.vehicle_instances = []
        self.total_num_vehicles = 0
        typeid_to_startidx = {}
        for type_config in self.vehicle_types:
            type_id = type_config['id']
            count = type_config['count']
            typeid_to_startidx[type_id] = self.total_num_vehicles
            for _ in range(count):
                instance_info = {
                    'type_id'         : type_id,                                                                        # Loại xe
                    'capacity'        : type_config['capacity'],                                                        # Capacity (kg)
                    'max_time'        : type_config['max_op_time_minutes'],                                             # Max Time (phút)
                    'fuel_cost_100km' : type_config['fuel_cost_100km'] / 100.0,                                         # Fuel Cost (VNĐ/100km)
                    'salary'          : type_config['driver_salary'],                                                   # Lương tài xế (VNĐ)
                    'reduction_factor': type_config['load_speed_reduction_factor'],                                     # Đô giảm tốc dựa trên tải
                }
                self.vehicle_instances.append(instance_info)
            self.total_num_vehicles += count

        # 1.1.3. Khởi tạo model (GNN - Context Embedding - Obs)
        self.node_embeddings = node_embeddings.cpu()
        self.context_embedder = context_embedder.cpu()
        self.embedding_dim = self.node_embeddings.shape[1]
        self.device = torch.device("cpu")
        _temp_builder = GNNFeature(config=self.config)
        _temp_data = _temp_builder.build()

        self.edge_data = self._preprocess_edge_data(_temp_data.edge_index, _temp_data.edge_attr)                        # Dictionary (i, j):[distance, travel_time_type1, fuel_cost_type1, travel_time_type2, fuel_cost_type2]
        self.observation_space = spaces.Dict({
            "context"         : spaces.Box(low=-np.inf, high=np.inf, shape=(self.embedding_dim,), dtype=np.float32),    # Context Emb
            "mask"            : spaces.Box(low=0, high=1, shape=(self.num_nodes,), dtype=np.int8),                      # Mask hiện tại
            "current_node_emb": spaces.Box(low=-np.inf, high=np.inf, shape=(self.embedding_dim,), dtype=np.float32),    # Emb hiện tại
            "vehicle_state"   : spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)                                 # [load/cap, time/max_time]
        })
        self.action_space = spaces.Discrete(self.num_nodes)

        # 1.1.4. Khởi tạo trạng thái động
        self.current_vehicle_idx     = None                                                                             # ID xe
        self.global_visited_mask     = np.zeros(self.num_nodes, dtype=bool)                                             # Mask toàn cục
        self.vehicle_states          = []                                                                               # Trạng thái xe
        self.all_routes              = []                                                                               # List chứa các Routes (mỗi route là list các node_id)
        self.current_step_in_episode = 0                                                                                # Đếm số bước thực hiện trong Episode                    
        self.total_episode_reward    = 0.0

        # 1.1.5. Thông số thưởng
        self.REWARD_DEMAND_FACTOR    = 250                                                                              # (x Demand) Thưởng dựa trên Demand
        self.REWARD_FUEL_FACTOR      = 2                                                                                # (x Fuel  ) Phạt dựa trên Fuel
        self.REWARD_REMAIN_LOAD      = 2000                                                                             # (x RemainLoad) Phạt dựa trên RemainLoad

    # --- 1.2. Hàm reset espisode ---
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.available_vehicle_indices = list(range(self.total_num_vehicles))
        self.current_vehicle_idx = np.random.choice(self.available_vehicle_indices)
        self.available_vehicle_indices.remove(self.current_vehicle_idx)
        self.global_visited_mask = np.zeros(self.num_nodes, dtype=bool)
        self.global_visited_mask[self.depot_idx] = True
        self.vehicle_states = []
        for _ in range(self.total_num_vehicles):
            self.vehicle_states.append({
                'current_node': self.depot_idx,
                'current_load': 0.0,
                'current_time': 0.0,
                'route': [self.depot_idx]
            })
        self.all_routes = [[] for _ in range(self.total_num_vehicles)]
        self.current_step_in_episode = 0
        self.total_episode_reward = 0.0
        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    # --- 1.3. Hàm thực hiện hành động mới ---
    def step(self, action):
        if self.current_vehicle_idx >= self.total_num_vehicles:
            terminated = True
            truncated = False
            reward = 0.0
            observation = self._get_observation()
            info = self._get_info()
            info['warning'] = "Stepped into finished environment"

            return observation, reward, terminated, truncated, info

        # 1.3.1. Trạng thái xe hiện tại
        vehicle_k = self.current_vehicle_idx                                                                            # ID xe
        state_k = self.vehicle_states[vehicle_k]                                                                        # Trạng thái
        prev_node = state_k['current_node']
        prev_load = state_k['current_load']

        # 1.3.2. Thông tin của action
        actual_travel_time = self._get_actual_travel_time(prev_node, action, prev_load, vehicle_k)                      # Thời gian (load) đến node tiếp
        _, _, fuel_cost_step = self._get_edge_info_for_vehicle(prev_node, action, vehicle_k)                            # Chi phí fuel đến node tiếp
        service_time_step = self.service_time[action] if action != self.depot_idx else 0.0                              # Thời gian service đến node tiếp
        demand_action = self.demand[action] if action != self.depot_idx else 0.0                                        # Nhu cầu của node tiếp
        vehicle_info_k = self.vehicle_instances[vehicle_k]
    
        # 1.3.3. Cập nhật kết quả của action
        state_k['current_node'] = action
        state_k['current_time'] += actual_travel_time + service_time_step
        state_k['current_load'] += demand_action
        state_k['route'].append(action)
        if action != self.depot_idx:                                                                                    # Cập nhật trạng thái toàn cục
            self.global_visited_mask[action] = True
        self.current_step_in_episode += 1

        # 1.3.4. Tính Reward
        reward = self._calculate_reward(fuel_cost_step, demand_action, state_k, vehicle_info_k)
        self.total_episode_reward += reward

        # 1.3.5. Kiểm tra kết thúc episode
        vehicle_done = False
        truncated = False
        terminated = False
        next_valid_mask = self._get_valid_mask()
        can_move_further = np.any(next_valid_mask)

        if action == self.depot_idx or not can_move_further:
            if action == self.depot_idx and len(state_k['route']) <= 2 and self.total_num_vehicles > 1 :
                vehicle_done = True
            elif len(state_k['route']) > 2 :
                vehicle_done = True
            if vehicle_done :
                self.all_routes[vehicle_k] = state_k['route']
                if self.available_vehicle_indices:
                    self.current_vehicle_idx = np.random.choice(self.available_vehicle_indices)
                    self.available_vehicle_indices.remove(self.current_vehicle_idx)
                else:
                    self.current_vehicle_idx = self.total_num_vehicles
                    terminated = True

        observation = self._get_observation()
        info = self._get_info()
        if terminated:
            info['final_cost'] = self.calculate_total_cost()
            info['all_routes'] = copy.deepcopy(self.all_routes)
            info['total_reward'] = self.total_episode_reward

        return observation, reward, terminated, truncated, info



    # --- 1.4. Hàm lấy tầm nhìn ---
    def _get_observation(self):
        if self.current_vehicle_idx >= self.total_num_vehicles:
            context_emb      = torch.zeros(self.embedding_dim, dtype=torch.float32, device=self.device)
            mask_np_bool     = np.zeros(self.num_nodes, dtype=bool)
            mask             = mask_np_bool.astype(np.int8)
            current_node_emb = torch.zeros(self.embedding_dim, dtype=torch.float32, device=self.device)
            vehicle_state    = np.array([1.0, 1.0], dtype=np.float32)
        else:
            vehicle_k        = self.current_vehicle_idx
            state_k          = self.vehicle_states[vehicle_k]
            vehicle_info_k   = self.vehicle_instances[vehicle_k]
            current_node     = state_k['current_node']
            current_load     = state_k['current_load']
            current_time     = state_k['current_time']
            current_capacity = vehicle_info_k['capacity']
            current_max_time = vehicle_info_k['max_time']
            epsilon = 1e-6

            # 1.4.1. Context Embedding (chạy trên CPU)
            with torch.no_grad():
                context_emb = self.context_embedder(
                    current_load             = current_load,
                    current_vehicle_capacity = current_capacity,
                    current_vehicle_max_time = current_max_time,
                    current_time             = current_time,
                    all_node_embeddings      = self.node_embeddings,
                    current_node_idx         = current_node,
                    visited_mask             = torch.tensor(self.global_visited_mask, dtype=torch.bool, device=self.device),
                    depot_idx                = self.depot_idx
                ).squeeze(0)

            # 1.4.2. Valid Mask
            mask_np_bool = self._get_valid_mask()
            mask = mask_np_bool.astype(np.int8) # Chuyển bool sang 0/1

            # 1.4.3. Current Emb
            current_node_emb = self.node_embeddings[current_node]

            # 1.4.4. State xe
            norm_load = min(current_load / (current_capacity + epsilon), 1.0)
            norm_time = min(current_time / (current_max_time + epsilon), 1.0)
            vehicle_state = np.array([norm_load, norm_time], dtype=np.float32)

        # Chuyển tensors sang NumPy arrays
        obs = {
            "context"         : context_emb.cpu().numpy(),
            "mask"            : mask,
            "current_node_emb": current_node_emb.cpu().numpy(),
            "vehicle_state"   : vehicle_state
        }

        return obs

    # --- 1.5. Hàm lấy thông tin ---
    def _get_info(self):
        info = {
            'current_vehicle': self.current_vehicle_idx,
            'current_step': self.current_step_in_episode,
            'current_total_reward': self.total_episode_reward
        }
        if self.current_vehicle_idx < self.total_num_vehicles:
            state_k = self.vehicle_states[self.current_vehicle_idx]
            info['vehicle_time'] = state_k['current_time']
            info['vehicle_load'] = state_k['current_load']
            info['vehicle_route'] = state_k['route'][-5:]

        return info

    # --- 1.6. Hàm lấy dữ liệu cạnh ---
    def _preprocess_edge_data(self, edge_index, edge_attr):
        edge_data_dict = {}
        edge_index_np = edge_index.cpu().numpy().T                                                                      # Chuyển sang [[i, j], [i, k], ...]
        edge_attr_np = edge_attr.cpu().numpy()                                                                          # Chuyển sang [distance, travel_time_type0, fuel_cost_type0, travel_time_type1, fuel_cost_type1]
        for (i, j), attr in zip(edge_index_np, edge_attr_np):                                                           # [i, j] [distance, travel_time_type0, fuel_cost_type0, travel_time_type1, fuel_cost_type1]
            edge_data_dict[(i, j)] = attr.tolist()

        return edge_data_dict

    # --- 1.7. Hàm lấy thông tin xe ---
    def _get_edge_info_for_vehicle(self, u, v, vehicle_idx):
        if u == v:
            return 0.0, 0.0, 0.0
        edge_features = self.edge_data.get((u, v))
        if edge_features is None:
            return float('inf'), float('inf'), float('inf')

        distance = edge_features[0]
        vehicle_type_id = self.vehicle_instances[vehicle_idx]['type_id']
        time_idx = 1 + vehicle_type_id * 2
        fuel_idx = time_idx + 1
        base_travel_time = edge_features[time_idx]
        fuel_cost_edge = edge_features[fuel_idx]
        return distance, base_travel_time, fuel_cost_edge

    # --- 1.8. Hàm thời gian di chuyển thực tế ---
    def _get_actual_travel_time(self, u, v, current_load, vehicle_idx):
        if u == v: return 0.0
        _, base_travel_time, _ = self._get_edge_info_for_vehicle(u, v, vehicle_idx)
        if base_travel_time == float('inf'): return float('inf')
        if base_travel_time <= 1e-9: return 0.0

        vehicle_info = self.vehicle_instances[vehicle_idx]
        vehicle_cap = vehicle_info['capacity']
        reduction_factor = vehicle_info['reduction_factor']
        effective_load = max(0.0, min(current_load, vehicle_cap)) 
        load_ratio = effective_load / vehicle_cap if vehicle_cap > 0 else 0.0
        speed_multiplier = (1.0 - reduction_factor * load_ratio)
        if speed_multiplier <= 1e-6:
            return float('inf') 
        actual_travel_time = base_travel_time / speed_multiplier

        return actual_travel_time

    # --- 1.9. Hàm tính thưởng ---
    def _calculate_reward(self, fuel_cost, demand, state_k, vehicle_info_k):
        visit_reward   = self.REWARD_DEMAND_FACTOR * demand
        fuel_penalty   = self.REWARD_FUEL_FACTOR * fuel_cost
        demand_penalty = self.REWARD_REMAIN_LOAD * (vehicle_info_k['capacity'] - state_k['current_load'])

        reward = visit_reward - fuel_penalty - demand_penalty

        return float(reward)

    # --- 1.10. Hàm lấy mask hợp lệ ---
    def _get_valid_mask(self):
        valid_mask = np.zeros(self.num_nodes, dtype=bool)

        if self.current_vehicle_idx >= self.total_num_vehicles:
            return valid_mask

        # 1.10.1. Lấy thông tin xe hiện tại
        vehicle_k = self.current_vehicle_idx
        state_k = self.vehicle_states[vehicle_k]
        vehicle_info_k = self.vehicle_instances[vehicle_k]
        current_node = state_k['current_node']
        current_load = state_k['current_load']
        current_time = state_k['current_time']
        current_capacity = vehicle_info_k['capacity']
        current_max_time = vehicle_info_k['max_time']

        # 1.10.2. Lặp qua các nút tiềm năng tiếp theo
        num_valid_customer_nodes = 0

        # 1.10.2. Lặp qua các nút tiềm năng tiếp theo
        for next_node in range(self.num_nodes):
            if next_node == current_node or next_node == self.depot_idx:
                continue

            if not self.global_visited_mask[next_node]:
                demand_next = self.demand[next_node]
                if current_load + demand_next <= current_capacity:
                    service_next = self.service_time[next_node]
                    time_to_next = self._get_actual_travel_time(current_node, next_node, current_load, vehicle_k)
                    time_after_service = current_time + time_to_next + service_next
                    if time_after_service <= current_max_time:
                        load_after_next = current_load + demand_next
                        time_next_to_depot = self._get_actual_travel_time(next_node, self.depot_idx, load_after_next, vehicle_k)
                        if time_after_service + time_next_to_depot <= current_max_time:
                            num_valid_customer_nodes += 1
                            valid_mask[next_node] = True

        # 1.10.3. Nếu còn khách hàng
        if num_valid_customer_nodes > 0:
            valid_mask[self.depot_idx] = False

        # 1.10.4. Nếu không còn khách hàng
        else:
            if current_node != self.depot_idx:
                time_to_depot = self._get_actual_travel_time(current_node, self.depot_idx, current_load, vehicle_k)
                if current_time + time_to_depot <= current_max_time:
                    valid_mask[self.depot_idx] = True
            elif current_node == self.depot_idx:
                valid_mask[self.depot_idx] = True

        # 1.10.5. Nếu ở kho và không đi được chỗ khác
        if current_node == self.depot_idx and not np.any(valid_mask[self.depot_idx+1:]): # Nếu ở kho và không đi được chỗ khác
            valid_mask[self.depot_idx] = True

        return valid_mask

    # --- 1.11. Hàm tính tổng chi phí ---    
    def calculate_total_cost(self):
        total_cost = 0.0
        total_fuel_cost = 0.0
        total_driver_salary = 0.0
        total_penalty = 0.0
        num_vehicles_used = 0

        # 1.11.1 Tính total cost
        for k, route in enumerate(self.all_routes):
            if len(route) > 2:
                vehicle_info = self.vehicle_instances[k]
                total_driver_salary += vehicle_info['salary']
                num_vehicles_used += 1
                route_fuel_cost = 0.0
                for i in range(len(route) - 1):
                    u = route[i]
                    v = route[i + 1]
                    _, _, fuel_cost_edge = self._get_edge_info_for_vehicle(u, v, k)
                    if fuel_cost_edge != float('inf'):
                       route_fuel_cost += fuel_cost_edge
                    else:
                       route_fuel_cost += 1e9
                total_fuel_cost += route_fuel_cost
        unvisited_nodes = []
        for i in range(self.num_nodes):
            if i != self.depot_idx and not self.global_visited_mask[i]:
                total_penalty += self.penalty_per_unvisited_node[i]
                unvisited_nodes.append(i)
        total_cost = total_fuel_cost + total_driver_salary + total_penalty

        return float(total_cost)

# --- 1.12. Hàm tạo môi trường song song ---
def make_env(rank: int, seed: int = 0, config_params: dict = None, node_embeddings_cpu: torch.Tensor = None) -> Callable:
    def _init() -> gym.Env:
        embedding_dim = node_embeddings_cpu.shape[1]
        vehicle_feature_dim=3
        context_embedder = ContextEmbedding(embedding_dim, vehicle_feature_dim=vehicle_feature_dim)
        env = VRPEnvGym(
            config_params=config_params,                                                                                # Truyền input vào env
            node_embeddings=node_embeddings_cpu,                                                                        # Truyền embedding vào env
            context_embedder=context_embedder                                                                           # Truyền context embedder vào env
        )
        env.reset(seed=seed + rank)
        return env
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)

    return _init