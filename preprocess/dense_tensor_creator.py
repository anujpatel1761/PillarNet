import numpy as np
import random

def dense_tensor(augmented_lidar_points,P=12000,N=32):

    pillar_group = {}

    for point in augmented_lidar_points:

        x_cord = np.floor(point[0]/0.16)
        y_cord = np.floor(point[1]/0.16)

        pillar_id  = (x_cord,y_cord)

        if pillar_id in pillar_group:
            pillar_group[pillar_id].append(point)
        else:
            pillar_group[pillar_id] = [(point)]

    print(f"Created {len(pillar_group)} non-empty pillars")


    # Step2 limit the pillar values to the 12000
    pillar_ids = list(pillar_group.keys()) 

    if len(pillar_ids) > P:
        selected_pillar_ids = random.sample(pillar_ids,P)
    else:
        selected_pillar_ids = pillar_ids

    dense_tensor = np.zeros((P, N, 9), dtype=np.float32)
    pillar_coords = np.zeros((P, 3), dtype=np.int32)

    for i, pillar_id in enumerate(selected_pillar_ids): 
        points_in_pillar = pillar_group[pillar_id]
        
        # Handle too many points per pillar (if > N)
        if len(points_in_pillar) > N:
            print(f"Pillar {pillar_id}: {len(points_in_pillar)} points, sampling {N}")
            sampled_points = random.sample(points_in_pillar, N)
        else:
            sampled_points = points_in_pillar

        for j, point in enumerate(sampled_points):
            dense_tensor[i, j, :] = point  # Fill with 9D features
        pillar_coords[i] = [0, int(pillar_id[1]), int(pillar_id[0])]

        if i < 3:
            print(f"Pillar {i}: ID={pillar_id}, Points={len(sampled_points)}")

    filled_pillars = len(selected_pillar_ids)
    empty_pillars = P - filled_pillars

    print(f"\n=== DENSE TENSOR STATISTICS ===")
    print(f"Total pillars: {P}")
    print(f"Filled pillars: {filled_pillars}")
    print(f"Empty pillars (padding): {empty_pillars}")
    print(f"Points per pillar limit: {N}")
    print(f"Dense tensor shape: {dense_tensor.shape}")
    print(f"Pillar coordinates shape: {pillar_coords.shape}")

    return dense_tensor, pillar_coords