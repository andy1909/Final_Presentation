import numpy as np
import pandas as pd

class VRPInputConfig:
    def __init__(self, 
                excel_depot_store = r"C:\Users\ASUS\Desktop\AI\FullStackProMax\Uploads\input.xlsx",
                excel_matrix_distance = r"C:\Users\ASUS\Desktop\AI\FullStackProMax\Uploads\Distance_matrix.xlsx",
                vehicle_types_config=[
                    {'id': 0, 'count': 20, 'capacity': 1000, 'avg_velocity': 40, 'fuel_cost_100km': 171000, 'driver_salary': 2500000, 'load_speed_reduction_factor': 0.10, 'max_op_time_minutes': 8*60},
                    {'id': 1, 'count': 20, 'capacity': 2000, 'avg_velocity': 38, 'fuel_cost_100km': 175000, 'driver_salary': 2650000, 'load_speed_reduction_factor': 0.12, 'max_op_time_minutes': 8*60}
                ]):

        # --- 1. Tạo map ---
        dt_node = pd.read_excel(excel_depot_store)                                                              # Excel thông tin depot và kho
        dt_distance = pd.read_excel(excel_matrix_distance, index_col=0)                                         # Excel khoảng cách giữa các nodes
        self.num_nodes                    = len(dt_node)                                                        # Số cửa hàng : 99 - Kho : 1
        self.coords                       = dt_node[['Kinh độ', 'Vĩ độ']].to_numpy()                            # Toạ độ      : (Kinh độ, vĩ độ)
        self.demand                       = dt_node['Nhu cầu'].to_numpy()                                       # Demand      : (kg)
        self.service_time                 = dt_node['Phục vụ'].to_numpy()                                       # Service Time: (phút)
        self.penalty_per_unvisited_node   = dt_node['Tiền phạt'].to_numpy()                                     # Tiền phạt   : (VNĐ)
        self.distance                     = dt_distance.to_numpy()                                              # Khoảng cách : (km)
        self.depot_idx                    = 0                                                                   # ID          : Kho là 0
        self.demand[self.depot_idx]       = 0                                                                   # Demand      : Kho là 0
        self.service_time[self.depot_idx] = 0                                                                   # Service Time: Kho là 0

        # --- 2. Tạo xe ---
        self.vehicle_types = vehicle_types_config                                                               # Số loại xe
        self.noise_matrix_velocity = np.random.uniform(0.8, 1.2, size=(self.num_nodes, self.num_nodes))
        self.noise_avg_velocity_types = []
        for pos in vehicle_types_config:
            noise_avg_velocity = pos['avg_velocity'] * self.noise_matrix_velocity                               # Velocity NOI: (km/h)
            self.noise_avg_velocity_types.append(noise_avg_velocity)