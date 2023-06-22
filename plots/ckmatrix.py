from matplotlib import pyplot as plt
from matplotlib import colors
import numpy as np

arr1d = list(range(0,5))

arr1d = ['table', 'robot', 'tablet', 'elsewhere', 'unknown']

red = [0, 255, 0]
white = [255, 0, 0]
whiterow = np.array([white for _ in range(5)])

data = np.array([whiterow for _ in range(5)])

for i, row in enumerate(data):
    row[i] = red

fig, ax = plt.subplots()
ax.imshow(data, cmap=None)

data = np.array([list('abcde'), list('fghij'), list('klmno'),list('pqrst'), list('uvwxy')])
for (i, j), z in np.ndenumerate(data):
    ax.text(j, i, '{}'.format(z), ha='center', va='center', size=10)

ax.yaxis.set_label_position("right")
plt.ylabel('annotator 2', fontsize=12)
plt.xlabel('annotator 1', fontsize=12)
plt.xticks(np.arange(len(arr1d)), arr1d)
plt.yticks(np.arange(len(arr1d)), arr1d)

ax.tick_params(axis="x", bottom=False, top=True, labelbottom=False, labeltop=True)
fig.tight_layout() 
plt.savefig('ckmatrix.png', dpi=300)
plt.close()
