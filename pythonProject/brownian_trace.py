import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Simulated motion trace (example)
x = np.random.normal(size=200000).cumsum()
y = np.random.normal(size=200000).cumsum()

# Load and display the base image
# image = Image.open("input_image.png")
# plt.imshow(image, alpha=0.7)

# Overlay the Brownian motion trace
plt.plot(x, y, color='darkgreen', alpha=0.7, linewidth=.15)

# Save the result
plt.axis('off')  # Turn off axes for clean output
plt.savefig("brownian_trace_overlay1.png", bbox_inches='tight', pad_inches=0, dpi=300)
