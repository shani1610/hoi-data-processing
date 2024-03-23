import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def visualize_3d_skeleton(pose):
    # Reshape the pose data into a 2D array
    pose_3d = np.array(pose).reshape(-1, 3)

    # Define the connections between joints to form a skeleton
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # Example connections, adjust as needed
        (0, 5), (5, 6), (6, 7), (7, 8),
        # Add more connections for other joints
    ]

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the joints
    ax.scatter(pose_3d[:, 0], pose_3d[:, 1], pose_3d[:, 2], c='r', marker='o')

    # Plot the connections to form a skeleton
    for connection in connections:
        joint1 = connection[0]
        joint2 = connection[1]
        ax.plot([pose_3d[joint1, 0], pose_3d[joint2, 0]],
                [pose_3d[joint1, 1], pose_3d[joint2, 1]],
                [pose_3d[joint1, 2], pose_3d[joint2, 2]], c='b')

    ax.set_title('3D Human Pose with Skeleton Visualization')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    plt.show()

if __name__ == "__main__":
    # Load the JSON file
    with open("./data/behave_seq/Date01_Sub01_backpack_back/t0005.000/k0.mocap.json", 'r') as json_file:
        data = json.load(json_file)

    # Extract the pose information
    pose = data.get("pose", [])

    # Visualize the 3D pose with skeleton
    visualize_3d_skeleton(pose)