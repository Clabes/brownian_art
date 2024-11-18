import requests
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from io import BytesIO

# Step 1: Download the image
# url = "https://www.worldhistory.org/img/r/p/500x600/7341.jpg?v=1506582017"
# url = "https://www.myinterestingfacts.com/wp-content/uploads/2014/03/Han-Dynasty-Dragon.jpg"
# url = "https://image.invaluable.com/housePhotos/tgbowo/68/628368/H20986-L147689995.jpg"
ver = 3
# response = requests.get(url)
# image = Image.open(BytesIO(response.content))

image = Image.open("/Users/clbar/math/brownian_art/pythonProject/images/img.png")

# Step 2: Convert to grayscale
grayscale_image = image.convert("L")

# Step 3: Perform edge detection using OpenCV's Canny method
# Convert grayscale image to NumPy array
grayscale_array = np.array(grayscale_image)

# Apply Canny edge detection
edges = cv2.Canny(grayscale_array, threshold1=100, threshold2=200)

# Convert edges back to PIL Image
edge_image = Image.fromarray(edges)

# Step 4: Save and display the results
grayscale_image.save(f"han_dragon_grayscale{ver}.png")
edge_image.save(f"han_dragon_edges{ver}.png")

# Display the images
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(image)
axes[0].set_title("Original Image")
axes[0].axis("off")

axes[1].imshow(grayscale_image, cmap="gray")
axes[1].set_title("Grayscale Image")
axes[1].axis("off")

axes[2].imshow(edge_image, cmap="gray")
axes[2].set_title("Edge Detection")
axes[2].axis("off")

plt.tight_layout()
plt.show()
