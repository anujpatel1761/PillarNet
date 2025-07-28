from preprocess.create_pillars import load_lidar_file, grid_discretization
from preprocess.dense_tensor_creator import dense_tensor



points = load_lidar_file("C:/Users/anujp/Desktop/PillarNet/data/000000.bin")

# convert into 9d
augmented_lidar_points = grid_discretization(points)
# Now convert into pillars
dense_tensor = dense_tensor(augmented_lidar_points)

# Save to text file
with open("dense_tensor_result.txt", "w") as f:
    f.write(str(dense_tensor))

print("Output saved to output.txt")

print(dense_tensor)