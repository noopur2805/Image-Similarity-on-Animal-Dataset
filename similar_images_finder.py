#!/usr/bin/env python3

'''
To execute:

python similar_images_finder.py -q <input filepath>
OR
python similar_images_finder.py --query <input filepath>


Example:

python similar_images_finder.py -q images/0.jpg
'''


import pandas as pd
import cv2
import argparse
import math

from skimage.measure import compare_ssim
from skimage.metrics import structural_similarity

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
# %matplotlib inline

parser = argparse.ArgumentParser(prog='Similar Images')
parser.add_argument('-q', '--query', help='path to the query image file')
parser.add_argument('-n', '--num_images', help='number of images')
args = parser.parse_args()

query_image = args.query
num_imgs = int(args.num_images)

# query_image = 'images/0.jpg'

tsne_cluster = pd.read_csv('tsne_clusters.csv', usecols=['id', 'x', 'y', 'cluster'])

idx = int(query_image.split('/')[-1].split('.')[0])
cluster_idx = int(tsne_cluster[tsne_cluster['id']==idx]['cluster'])

# Get all images belonging to the cluster same as query image 
idx_list = tsne_cluster[tsne_cluster['cluster']==cluster_idx]['id']

# Structural similarity
print('Finding most similar images..')
similar_images = {}
for i in idx_list:
    i1 = cv2.imread(query_image)
    i2 = cv2.imread('images/'+str(i)+'.jpg')
    ssim = structural_similarity(i1, i2, multichannel=True)
    similar_images[i] = ssim
    # print(i)
    
    
import operator
most_similar_idx_ordered = dict(sorted(similar_images.items(), key=operator.itemgetter(1),reverse=True))
most_similar_imgs = list(most_similar_idx_ordered.keys())[1:10]
most_similar_imgs = ['images/'+str(i)+'.jpg' for i in most_similar_imgs]
import math

num=5


#!/usr/bin/env python3

'''
To execute:

python similar_images_finder.py -q <input filepath>
OR
python similar_images_finder.py --query <input filepath>


Example:

python similar_images_finder.py -q images/0.jpg
'''


import pandas as pd
import cv2
import argparse
import math

from skimage.measure import compare_ssim
from skimage.metrics import structural_similarity

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
# %matplotlib inline

parser = argparse.ArgumentParser(prog='Similar Images')
parser.add_argument('-q', '--query', help='path to the query image file')
parser.add_argument('-n', '--num_images', help='number of images')
args = parser.parse_args()

tsne_cluster = pd.read_csv('tsne_clusters.csv', usecols=['id', 'x', 'y', 'cluster'])

idx = int(query_image.split('/')[-1].split('.')[0])
cluster_idx = int(tsne_cluster[tsne_cluster['id']==idx]['cluster'])

# Get all images belonging to the cluster same as query image 
idx_list = tsne_cluster[tsne_cluster['cluster']==cluster_idx]['id']

# Structural similarity
print('Finding most similar images..')
similar_images = {}
for i in idx_list:
    i1 = cv2.imread(query_image)
    i2 = cv2.imread('images/'+str(i)+'.jpg')
    ssim = structural_similarity(i1, i2, multichannel=True)
    similar_images[i] = ssim
    # print(i)

    
import operator
most_similar_idx_ordered = dict(sorted(similar_images.items(), key=operator.itemgetter(1),reverse=True))
most_similar_imgs = list(most_similar_idx_ordered.keys())[1:num_imgs+1]
most_similar_imgs = ['images/'+str(i)+'.jpg' for i in most_similar_imgs]



fig = plt.figure(figsize=(10,10), constrained_layout=True)
num_rows = math.ceil(num_imgs/3)+1
num_cols = 3
gs = fig.add_gridspec(num_rows, num_cols)
ax = fig.add_subplot(gs[0,:1])
img = cv2.cvtColor(cv2.imread(query_image), cv2.COLOR_RGB2BGR)
ax.imshow(img)

row = 1
num_plot_flag = 3 # When plots are less than multiples of 3 

for i in range(len(most_similar_imgs), -1, -3):
    
    if i%3 == 2:
        num_plot_flag = 2
    elif i%3 == 1:
        num_plot_flag = 1
    else:
        pass

    ax1 = fig.add_subplot(gs[row,0])
    img1 = cv2.cvtColor(cv2.imread(most_similar_imgs[i-1]), cv2.COLOR_RGB2BGR)
    ax1.imshow(img1)

    if ((row==num_rows-1) and (num_plot_flag == 1)):
        break

    ax2 = fig.add_subplot(gs[row,1])
    img2 = cv2.cvtColor(cv2.imread(most_similar_imgs[i-2]), cv2.COLOR_RGB2BGR)
    ax2.imshow(img2)

    if ((row==num_rows-1) and (num_plot_flag == 2)):
        break

    ax3 = fig.add_subplot(gs[row,2])
    img3 = cv2.cvtColor(cv2.imread(most_similar_imgs[i-3]), cv2.COLOR_RGB2BGR)
    ax3.imshow(img3)

    row += 1


plt.show(fig)

import time
timestr = time.strftime("%Y-%m-%d_%H%M%S")
fig.savefig('figure_' + timestr + '.jpg')


