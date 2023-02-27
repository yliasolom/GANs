import os
import matplotlib.pyplot as plt
from scipy.misc import imresize

# my dir
root = 'data/celebA/celebA/'
save_root = 'data/resized_celebA/'
resize_size = 64

if not os.path.isdir(save_root):
    os.mkdir(save_root)
if not os.path.isdir(save_root + 'celebA'):
    os.mkdir(save_root + 'celebA')
img_list = os.listdir(root)



for i in range(len(img_list)):
    img = plt.imread(root + img_list[i])
    img = imresize(img, (resize_size, resize_size))
    plt.imsave(fname=save_root + 'celebA/' + img_list[i], arr=img)

  
