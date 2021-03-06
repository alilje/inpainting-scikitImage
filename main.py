from PIL import Image
import skimage.restoration.inpaint
import sklearn
import numpy as np
from matplotlib._cm_listed import cmaps
from skimage import data
from skimage.restoration import inpaint
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from collections import OrderedDict

file = '/Users/alilje/Desktop/cat_with_TEXT.jpg'

im_IMAGE = Image.open(file) # Open CAT JPG
print("im_IMAGE is of type " + str(type(im_IMAGE)))
print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")

im_ARRAY_I = np.array(im_IMAGE)
print("im_ARRAY_I is of type " + str(type(im_ARRAY_I)))
print("im_ARRAY_I element: " + str(im_ARRAY_I[0:1, 1]))
mask0_ARRAY_F = np.zeros(im_ARRAY_I.shape)
print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")

print("mask0_ARRAY_F is of type " + str(type(mask0_ARRAY_F)))
print("mask0_ARRAY_F element: " + str(mask0_ARRAY_F[0:1, 1]))
print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")

mask0_ARRAY_I = mask0_ARRAY_F.astype(np.uint8)
print("mask0_ARRAY_I is of type " + str(type(mask0_ARRAY_I)))
print("mask0_ARRAY_I element: " + str(mask0_ARRAY_I[0:1, 1]))
print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")


image_COPY_I = im_ARRAY_I.copy()
print("image_COPY_I is of type " + str(type(image_COPY_I)))
print("image_COPY_I element: " + str(image_COPY_I[0:1, 1]))
print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
# Add 1's to mask

mask0_ARRAY_I[60:70, 55:100] = 1
print("mask0_ARRAY_I is of type " + str(type(mask0_ARRAY_I)))
print("mask0_ARRAY_I 0and1 element: " + str(mask0_ARRAY_I[0:101, 0:800]))
print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
im_ARRAY_I[np.where(mask0_ARRAY_I)] = 0


image_result = inpaint.inpaint_biharmonic(im_ARRAY_I, mask0_ARRAY_I, multichannel=False)


cmaps['Sequential'] = [
    'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
    'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
    'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']

maps = cmaps['Sequential']
a = maps[0]
print(a)
fig, axes = plt.subplots(ncols=2, nrows=2)
ax = axes.ravel()

ax[0].set_title('Original image')
ax[0].imshow(im_IMAGE)

ax[1].set_title('Mask')
ax[1].imshow(mask0_ARRAY_I)

ax[2].set_title('Defected image')
ax[2].imshow(mask0_ARRAY_I + im_ARRAY_I)

ax[3].set_title('Inpainted image')
ax[3].imshow(image_result,cm.get_cmap('gray'))

for a in ax:
    a.axis('off')

fig.tight_layout()
plt.show()
