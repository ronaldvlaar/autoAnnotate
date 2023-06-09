import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

im = Image.open('frames/33001_sessie1_taskrobotEngagement.png')

# Create figure and axes
fig, ax = plt.subplots()

# Display the image
ax.imshow(im)

# Create a Rectangle patch
rect = patches.Rectangle((159, 143), 350-159, 460-143, linewidth=1, edgecolor='r', facecolor='none')

# Add the patch to the Axes
ax.add_patch(rect)

plt.show()
