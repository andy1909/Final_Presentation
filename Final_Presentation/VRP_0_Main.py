import os
import time
import torch
import json
import itertools
import openrouteservice
import pandas as pd
import numpy as np

from VRP_1_Input import VRPInputConfig
from VRP_2_Model import GNNFeature, ModelComponents, ContextEmbedding
from VRP_3_Env import VRPEnvGym, make_env
from VRP_4_PPO import VRPActorCriticPolicy
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

# === Train ===
TOTAL_TIMESTEPS_TRAIN = 20000

# === Đường dẫn file ===
BASE_UPLOADS_DIR = Path("Uploads")
EXCEL_FILE_PATH = BASE_UPLOADS_DIR / "input.xlsx"
DISTANCE_MATRIX_PATH = BASE_UPLOADS_DIR / "Distance_matrix.xlsx"
MODEL_SAVE_PATH = BASE_UPLOADS_DIR / "vrp_ppo_model.zip"
VRP_OUTPUT_PATH = BASE_UPLOADS_DIR / "data.json"

# === Thông số xe ===
VEHICLE_TYPES_CONFIG = [
    {'id': 0, 'count': 20, 'capacity': 1000, 'avg_velocity': 40, 'fuel_cost_100km': 171000, 'driver_salary': 2500000, 'load_speed_reduction_factor': 0.10, 'max_op_time_minutes': 8*60},
    {'id': 1, 'count': 20, 'capacity': 2000, 'avg_velocity': 38, 'fuel_cost_100km': 175000, 'driver_salary': 2650000, 'load_speed_reduction_factor': 0.12, 'max_op_time_minutes': 8*60}
]

# === Thông số model ===
CFG_GNN_HIDDEN_DIM = 1024
CFG_EMBEDDING_DIM = 64
CFG_DECODER_HIDDEN_DIM = 256
CFG_NUM_HEADS = 4
CFG_CRITIC_HIDDEN_DIM = 128
CFG_VEHICLE_FEATURE_DIM = 3

# === API key ===
ORS_API_KEY = '5b3ce3597851110001cf6248b00ac6acf0fc43ec93ea37239079ce59'

# === Seed ===
SEED = 30

def calculate_distance_matrix():
    dt_store = pd.read_excel(EXCEL_FILE_PATH)
    client = openrouteservice.Client(key = ORS_API_KEY)

    num_nodes = len(dt_store)
    coords = dt_store[['Kinh độ', 'Vĩ độ']].to_numpy()
    distance = np.zeros((num_nodes, num_nodes))
    for i, j in itertools.product(range(0, num_nodes, 59), repeat=2):
        sources = list(range(i, min(i + 59, num_nodes)))
        destinations = list(range(j, min(j + 59, num_nodes)))

        response = client.distance_matrix(
            locations=coords.tolist(),
            profile='driving-car',
            metrics=['distance'],
            units='km',
            sources=sources,
            destinations=destinations
        )
        distance[np.ix_(sources, destinations)] = np.array(response['distances'])

    df_distance = pd.DataFrame(
        distance,
        index=[f'Node {i}' for i in range(num_nodes)],
        columns=[f'Node {j}' for j in range(num_nodes)])
    df_distance.to_excel(DISTANCE_MATRIX_PATH)

def train_vrp_model():
    vrp_config_dict_for_env_train = {
        'excel_depot_store': EXCEL_FILE_PATH,
        'excel_matrix_distance': DISTANCE_MATRIX_PATH,
        'vehicle_types_config': VEHICLE_TYPES_CONFIG
    }
    config_for_gnn_train = VRPInputConfig(**vrp_config_dict_for_env_train)
    feature_builder_train = GNNFeature(config_for_gnn_train)
    data_train = feature_builder_train.build()

    model_suite_train = ModelComponents(
        data_GNN=data_train,
        feature_GNN=feature_builder_train,
        gnn_hidden_dim=CFG_GNN_HIDDEN_DIM,
        embedding_dim=CFG_EMBEDDING_DIM,
        num_heads=CFG_NUM_HEADS,
        decoder_hidden_dim=CFG_DECODER_HIDDEN_DIM,
        critic_hidden_dim=CFG_CRITIC_HIDDEN_DIM,
        vehicle_feature_dim=CFG_VEHICLE_FEATURE_DIM
    )
    gnn_encoder_train = model_suite_train.gnn_encoder
    gnn_encoder_train.eval()
    with torch.no_grad():
        node_embeddings_train = gnn_encoder_train(data_train)
    node_embeddings_cpu_train = node_embeddings_train.cpu()

    num_cpu_train = os.cpu_count()
    env_seed_train = SEED
    vec_env_train = SubprocVecEnv([make_env(i, env_seed_train, vrp_config_dict_for_env_train, node_embeddings_cpu_train) for i in range(num_cpu_train)], start_method='spawn')

    policy_kwargs_train = dict(
        node_embeddings=node_embeddings_cpu_train,
        embedding_dim=CFG_EMBEDDING_DIM,
        decoder_hidden_dim=CFG_DECODER_HIDDEN_DIM,
        critic_hidden_dim=CFG_CRITIC_HIDDEN_DIM,
    )

    effective_n_steps_train = max(1, 2048 // num_cpu_train)

    ppo_config_train = {
        "policy": VRPActorCriticPolicy,
        "env": vec_env_train,
        "policy_kwargs": policy_kwargs_train,
        "learning_rate": 3e-4,
        "n_steps": effective_n_steps_train,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "verbose": 1,
        "tensorboard_log": "./vrp_ppo_tensorboard/",
        "device": "auto",
    }

    model_train = PPO(**ppo_config_train)
    total_timesteps_train = TOTAL_TIMESTEPS_TRAIN
    model_train.learn(total_timesteps=total_timesteps_train, log_interval=1)
    model_train.save(MODEL_SAVE_PATH)
    vec_env_train.close()

def calculate_total_cost(routes, eval_env):
    distance_matrix_df = pd.read_excel(DISTANCE_MATRIX_PATH, index_col=0)
    distance_matrix = distance_matrix_df.to_numpy()
    total_cost = 0.0

    for i, route in enumerate(routes):
        route_distance = 0.0
        for j in range(len(route) - 1):
            route_distance += distance_matrix[route[j], route[j + 1]]
        route_distance += distance_matrix[route[-1], route[0]]
        fuel_cost = route_distance * eval_env.vehicle_instances[i]['fuel_cost_100km'] / 100
        total_cost += fuel_cost + eval_env.vehicle_instances[i]['salary']

    return total_cost

def run_vrp_inference():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    vrp_config_dict_for_env_infer = {
        'excel_depot_store': EXCEL_FILE_PATH,
        'excel_matrix_distance': DISTANCE_MATRIX_PATH,
        'vehicle_types_config': VEHICLE_TYPES_CONFIG
    }

    config_for_gnn_infer = VRPInputConfig(**vrp_config_dict_for_env_infer)
    feature_builder_infer = GNNFeature(config_for_gnn_infer)
    data_infer = feature_builder_infer.build()

    model_suite_infer = ModelComponents(
        data_GNN=data_infer,
        feature_GNN=feature_builder_infer,
        gnn_hidden_dim=CFG_GNN_HIDDEN_DIM,
        embedding_dim=CFG_EMBEDDING_DIM,
        num_heads=CFG_NUM_HEADS,
        decoder_hidden_dim=CFG_DECODER_HIDDEN_DIM,
        critic_hidden_dim=CFG_CRITIC_HIDDEN_DIM,
        vehicle_feature_dim=CFG_VEHICLE_FEATURE_DIM
    )

    gnn_encoder_infer = model_suite_infer.gnn_encoder
    gnn_encoder_infer.eval()
    with torch.no_grad():
        node_embeddings_infer = gnn_encoder_infer(data_infer)
    node_embeddings_cpu_infer = node_embeddings_infer.cpu()

    context_embedder_eval = ContextEmbedding(
        embedding_dim=CFG_EMBEDDING_DIM,
        vehicle_feature_dim=CFG_VEHICLE_FEATURE_DIM
    )

    eval_env = VRPEnvGym(
        config_params=vrp_config_dict_for_env_infer, 
        node_embeddings=node_embeddings_cpu_infer,
        context_embedder=context_embedder_eval
    )

    custom_objects_infer = {
        "policy_kwargs": dict(
            node_embeddings=node_embeddings_cpu_infer,
            embedding_dim=CFG_EMBEDDING_DIM,
            decoder_hidden_dim=CFG_DECODER_HIDDEN_DIM,
            critic_hidden_dim=CFG_CRITIC_HIDDEN_DIM,
        )
    }

    loaded_model_infer = PPO.load(
        MODEL_SAVE_PATH,
        env=eval_env,
        policy=VRPActorCriticPolicy,
        custom_objects=custom_objects_infer
    )

    obs_tuple_infer = eval_env.reset(seed=SEED)

    if isinstance(obs_tuple_infer, tuple) and len(obs_tuple_infer) == 2:
        obs_infer, info_infer = obs_tuple_infer
    else:
        obs_infer = obs_tuple_infer
        info_infer = {}

    terminated_infer = False
    truncated_infer = False
    total_reward_eval_infer = 0
    episode_steps_infer = 0

    while not terminated_infer and not truncated_infer:
        action_output_infer, _states_infer = loaded_model_infer.predict(obs_infer, deterministic=True)

        action_infer = int(action_output_infer.item())

        step_result_infer = eval_env.step(action_infer)
        obs_infer, reward_infer, terminated_infer, truncated_infer, info_infer = step_result_infer

        total_reward_eval_infer += reward_infer
        episode_steps_infer += 1

    final_cost_infer = info_infer.get('final_cost', "N/A")
    all_routes_infer = info_infer.get('all_routes', [])
    valid_routes_for_json = []
    r = 0
    pernalty_per_unvisited_node = eval_env.penalty_per_unvisited_node
    unvisited_node = [node for node in range(eval_env.num_nodes) if node not in itertools.chain.from_iterable(all_routes_infer)]

    if all_routes_infer:
        for i, route in enumerate(all_routes_infer):
            if route and len(route) > 1 and not (len(route) <=2 and route[0] == eval_env.depot_idx and route[-1] == eval_env.depot_idx):
                r += 1
                route_demand = sum(float(eval_env.demand[node]) for node in route if node != eval_env.depot_idx)
                route_time = sum(float(eval_env.service_time[node]) for node in route if node != eval_env.depot_idx)
                route_cost = calculate_total_cost([route], eval_env)

                print(f"Vehicle {r} (Type {eval_env.vehicle_instances[i]['type_id']}): {route}")
                print(f"    - Total Demand: {route_demand:.2f} kg")
                print(f"    - Total Service Time: {route_time:.2f} minutes")
                print(f"    - Total Route Cost: {route_cost:.2f} VND")
                # Adding vehicle type to JSON output
                valid_routes_for_json.append({
                    "id": int(r),
                    "cost": float(route_cost),  # Convert to Python float
                    "time": float(route_time),    # Convert to Python float
                    "type": int(eval_env.vehicle_instances[i]['type_id']),  # Add vehicle type (0 or 1)
                    "route": [int(node) for node in route]  # Convert numpy.int64 to Python int
                })
    final_cost_infer += sum(pernalty_per_unvisited_node[node] for node in unvisited_node) if unvisited_node else 0
    print(f"Total Cost: {final_cost_infer:.2f} VND")

    json_output_path = VRP_OUTPUT_PATH
    try:
        # Writing JSON with each route object on a single line for compact formatting
        with open(json_output_path, 'w', encoding='utf-8') as f_json:
            f_json.write("[\n")
            for i, route_data in enumerate(valid_routes_for_json):
                # Serialize each route object to a single-line string
                route_json = json.dumps(route_data, ensure_ascii=False)
                f_json.write(f"  {route_json}")
                if i < len(valid_routes_for_json) - 1:
                    f_json.write(",\n")
                else:
                    f_json.write("\n")
            f_json.write("]\n")
        print(f"\nSuccessfully saved routes to {json_output_path}")
    except Exception as e:
        print(f"\nError saving routes to JSON: {e}")

    eval_env.close()

if __name__ == "__main__":
    overall_start_time = time.time()

    calculate_distance_matrix()

    train_vrp_model()

    run_vrp_inference()

    print(f"--- TOTAL TIME {time.time() - overall_start_time:.2f} seconds ---")
