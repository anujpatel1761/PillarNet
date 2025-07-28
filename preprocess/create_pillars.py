import numpy as np

def load_lidar_file(file_path):
    points = np.fromfile(file_path, dtype=np.float32).reshape(-1,4)
    # For testing take only first 10 points 
    points = points
    return points



def grid_discretization(lidar_points,grid_resolution_x=0.16,grid_resolution_y=0.16):

    # Step 1: Group points into pillars using grid discretization
    # Formula: pillar_index = floor(coordinate / grid_resolution)
    group_points_by_pillars = {}

    for point in lidar_points:
        # Discretize x,y coordinates into grid cells
        pillar_x = np.floor(point[0]/grid_resolution_x)  # Which pillar in x direction
        pillar_y = np.floor(point[1]/grid_resolution_y)  # Which pillar in y direction
        pillar_id = (pillar_x,pillar_y)                  # Unique pillar identifier
        
        # Group points that fall in the same pillar
        if pillar_id in group_points_by_pillars:
            group_points_by_pillars[pillar_id].append((point[0],point[1],point[2],point[3]))
        else:
            group_points_by_pillars[pillar_id] = [(point[0],point[1],point[2],point[3])]

    # Step 2: Calculate pillar centers (geometric center of each grid cell)
    # Formula: center = pillar_index × grid_resolution + (grid_resolution / 2)
    calculate_pillar_centers = {}
    for index,value in group_points_by_pillars.items():
        calculate_center_x = index[0] * grid_resolution_x + (grid_resolution_x/2)
        calculate_center_y = index[1] * grid_resolution_y + (grid_resolution_y/2)
        calculate_pillar_centers[index] = [(calculate_center_x,calculate_center_y)]

    # Step 3: Calculate arithmetic mean of all points in each pillar
    # Formula: mean = sum_of_coordinates / number_of_points
    calculate_arithmetic_mean_for_each_pillar = {}

    for index,value in group_points_by_pillars.items():
        sum_x = 0 
        sum_y = 0
        sum_z = 0

        # Sum all coordinates in this pillar
        for i in range(len(value)):
            sum_x += value[i][0]
            sum_y += value[i][1]
            sum_z += value[i][2]
        
        # Calculate average coordinates (cluster center)
        num_points = len(value)
        mean_x = sum_x / num_points 
        mean_y = sum_y / num_points
        mean_z = sum_z / num_points
        calculate_arithmetic_mean_for_each_pillar[index] = [(mean_x,mean_y,mean_z)]

    # Step 4: Create 9D augmented features for each point
    # Original: [x, y, z, intensity] → Augmented: [x, y, z, r, xc, yc, zc, xp, yp]
    points = []
    for key,value in group_points_by_pillars.items():
        for j in range(len(value)):
            if (key in calculate_arithmetic_mean_for_each_pillar) and (key in calculate_pillar_centers):
                result = calculate_arithmetic_mean_for_each_pillar[key]        # Cluster center
                center_relative = calculate_pillar_centers[key]               # Pillar center

                # Distance from point to cluster center (c = cluster)
                xc = value[j][0] - result[0][0]      # x_point - x_cluster_mean
                yc = value[j][1] - result[0][1]      # y_point - y_cluster_mean  
                zc = value[j][2] - result[0][2]      # z_point - z_cluster_mean
                
                # Distance from point to pillar center (p = pillar)
                xp = value[j][0] - center_relative[0][0]  # x_point - x_pillar_center
                yp = value[j][1] - center_relative[0][1]  # y_point - y_pillar_center
                
                # Combine into 9D feature: [x, y, z, intensity, xc, yc, zc, xp, yp]
                point = value[j][0],value[j][1],value[j][2],value[j][3],xc,yc,zc,xp,yp
                points.append(point)

    return points
