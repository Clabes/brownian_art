import cv2
from PIL import Image
from scipy.ndimage import distance_transform_edt
import numpy as np
import matplotlib.pyplot as plt

def exp_potential_well(beta = 1):
    # Load the edge-detected image
    # ver = 2
    edge_image = Image.open("han_dragon_edges3.png")  # Replace with your file path

    # Convert to a NumPy array
    edge_array = np.array(edge_image)


    # Normalize the edge array to binary (0 for background, 1 for edges)
    edge_binary = (edge_array > 0).astype(np.uint8)



    # Calculate distance transform (distance to the nearest edge pixel)
    distance_map = distance_transform_edt(1 - edge_binary)  # Invert binary for proper distance calculation


    # Create a potential well: Exponentially decay the potential based on the distance map
    # Higher potential far from edges, lower near edges
    potential_well = np.exp(-beta*distance_map / np.max(distance_map))

    # Visualize the potential well
    plt.figure(figsize=(8, 6))
    plt.imshow(potential_well, cmap="hot", origin="upper")
    plt.colorbar(label="Potential")
    plt.title("Potential Well Based on Edge Detection")
    plt.axis("off")
    plt.show()


    # Save the potential well as a .npy file
    np.save(f"images/exp_potential/potential_well_{beta}.npy", potential_well)


def blur_potential_well(beta = 10):
    # Create a potential field where edges have low potential

    edge_image = Image.open("han_dragon_grayscale3.png")  # Replace with your file path
    edges = np.array(edge_image)

    potential_field = np.where(edges > 0, 0, 255)  # Low potential (0) at edges

    potential_field = np.array(edge_image.convert("L"))

    potential_field = cv2.GaussianBlur(potential_field, (15, 15), beta)  # Smooth the field

    # Normalize the potential field to [0, 1]
    potential_field = 1 - (potential_field / 255.0)

    # Visualize the potential well
    plt.imshow(potential_field, cmap="viridis")
    plt.colorbar(label="Potential (Low -> High)")
    plt.title("Blur Potential Well")
    plt.axis("off")
    plt.savefig(f"blur_potential_well_{beta}.png")
    plt.show()


if __name__ == '__main__':
    blur_potential_well()
    # potential_well(beta = k)
    # plt.clear()