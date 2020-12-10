#!/usr/bin/env python3

"""Preprocess images using Keras pre-trained models."""

import argparse
import csv
import os
import csv

from keras import applications
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing import image

from sklearn import manifold
from glob import glob

import numpy as np
import pandas
import keras


parser = argparse.ArgumentParser(prog='Feature extractor')
parser.add_argument('source', default=None, help='Path to the source file')
pargs = parser.parse_args()

source_dir = os.path.dirname(pargs.source)
model = applications.resnet50.ResNet50(weights='imagenet', include_top=False, pooling='avg')

import re
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def change_model(model, new_input_shape=(None, 512, 512, 3)):
    # To change the input size of model to match with current image size
    
    model._layers[1].batch_input_shape = new_input_shape
    model._layers[2].pool_size = (8, 8)
    model._layers[2].strides = (8, 8)

    new_model = keras.models.model_from_json(model.to_json())

    # copying weights from old model to new one
    for layer in new_model.layers:
        try:
            layer.set_weights(model.get_layer(name=layer.name).get_weights())
        except:
            print("Could not transfer weights for layer {}".format(layer.name))

    return new_model

def get_feature(img_path):

    print('file: {}'.format(img_path))
    try:
        # setting the image size to 224 x 224 
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # model = change_model(model)
        features = model.predict(x)[0]
        features_arr = np.char.mod('%f', features)
        
        return features_arr
    
    except Exception as ex:
        print(ex)
        pass
    return None

def process(data):
    
    ids = list(range(len(data)))

    x_data = np.asarray(data).astype('float64')
    x_data = x_data.reshape((x_data.shape[0], -1))

    # perform t-SNE
    model = manifold.TSNE(random_state=0)
    vis_data = model.fit_transform(x_data)

    # convert the results into a list of dict of (x,y) coordinates of points
    results = []
    for i in range(0, len(data)):
        results.append({
            'id': ids[i],
            'x': vis_data[i][0],
            'y': vis_data[i][1]
        })
    return results

def run():
    try:

        # extract features
        features = [get_feature(i) for i in sorted(glob(source_dir + '/*.jpg'), key=numericalSort)]
        results = process(features)
        return results

    except EnvironmentError as e:
        print(e)

def write_csv(data):
    
    csv_columns = ['id','x','y']
    csv_file = "tsne_features.csv"
    
    try:
        with open(csv_file, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for data in data:
                writer.writerow(data)
    except IOError:
        print("I/O error")
    
    
if __name__ == '__main__':
    result = run()
    write_csv(result)
