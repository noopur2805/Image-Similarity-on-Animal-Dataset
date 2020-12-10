FINDING  SIMILAR  IMAGES


    1. tsne_feature_extractor.py
This file takes image as input and using ResNet50 extracts feature vectors. These feature vectors are used to get TSNE coordinates, which are used to get clusters.	
 
    2. clustering.ipynb
In this notebook, clusters were found out using Kmeans and DBSCAN. Mainly DBSCAN was used, and Kmeans validated the clusters obtained using the former.

    3. similar_images_finder.py
Takes query image as input and outputs the most similar images obtained by “structural similarity” score.


'''
To execute:

python similar_images_finder.py -q <input filepath> -n <number of images>
OR
python similar_images_finder.py --query <input filepath> --num_images <number of images>


Example:

python similar_images_finder.py -q images/0.jpg -n 9
'''

