from keras.models import Sequential, Model, load_model,model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau
from keras.optimizers import SGD,adagrad,adadelta,adam
from keras.regularizers import l2
import tkinter
import math
from keras.applications.inception_resnet_v2 import preprocess_input
from keras.utils import multi_gpu_model
import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
from scipy.misc import imresize
from collections import defaultdict
import collections
import json
import os
import ast
import urllib.request

class_to_ix = {}
ix_to_class = {}
#chuyen file classes.txt thanh mang chua cac class
link='~/CNN/food-101/'
with open(link+'meta/classes.txt', 'r') as txt:
    classes = [l.strip() for l in txt.readlines()]
    class_to_ix = dict(zip(classes, range(len(classes))))
    ix_to_class = dict(zip(range(len(classes)), classes))
    class_to_ix = {v: k for k, v in ix_to_class.items()}
sorted_class_to_ix = collections.OrderedDict(sorted(class_to_ix.items()))

model = load_model(filepath='~/CNN/saved/Food101/trained model/train3/model_InceptionV3.29-0.71-0.83.hdf5')
# model = load_model(filepath=link+'trained model/model_inceptionV3.17-0.65-0.85.hdf5')
# with open(link+'trained model/json/model.json','r') as f:
#     model=model_from_json(f.read())

# model = multi_gpu_model(model,gpus=2)
# model.load_weights(link+'trained model/json/model.h5')
def generate_dir_file_map(path):
    dir_files = []
    dir_classes = []
    with open(path, 'r') as txt:
        files = [l.strip() for l in txt.readlines()]
        for f in files:
            dir_name, id = f.split('/')
            dir_files.append(id+'.jpg')
            dir_classes.append(dir_name)
    return dir_files, dir_classes


test_dir_files,test_dir_classes = generate_dir_file_map(link+'meta/test.txt')

def center_crop(x, center_crop_size, **kwargs):
    centerw, centerh = x.shape[0] // 2, x.shape[1] // 2
    halfw, halfh = center_crop_size[0] // 2, center_crop_size[1] // 2
    return x[centerw - halfw:centerw + halfw + 1, centerh - halfh:centerh + halfh + 1, :]


def predict_10_crop(img, top_n=5,debug=False):
    flipped_X = np.fliplr(img)
    crops = [
        img[:299, :299, :],  # Upper Left
        img[:299, img.shape[1] - 299:, :],  # Upper Right
        img[img.shape[0] - 299:, :299, :],  # Lower Left
        img[img.shape[0] - 299:, img.shape[1] - 299:, :],  # Lower Right
        center_crop(img, (299, 299)),

        flipped_X[:299, :299, :],
        flipped_X[:299, flipped_X.shape[1] - 299:, :],
        flipped_X[flipped_X.shape[0] - 299:, :299, :],
        flipped_X[flipped_X.shape[0] - 299:, flipped_X.shape[1] - 299:, :],
        center_crop(flipped_X, (299, 299))
    ]

    y_pred = model.predict(np.array(crops))
    preds = np.argmax(y_pred, axis=1)
    top_n_preds = np.argpartition(y_pred, -top_n)[:, -top_n:]
    if debug:
        print('Top-1 Predicted:', preds)
        print('Top-5 Predicted:', top_n_preds)
    return preds, top_n_preds

# ix = 13001
# test_datagen = ImageDataGenerator()
# test_generator = test_datagen.flow_from_directory(link+'test',
#         target_size=(299, 299),
#         batch_size=64,
#         class_mode='categorical',seed=11)

# opt = adam(lr=0.0001,beta_1=0.9,beta_2=0.999,epsilon=1e-08,decay=0.0)#thuat toan toi uu
# model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])#compile model
# scores=model.evaluate_generator(test_generator,test_generator.n/64,verbose=1)#testing
# print('%s: %.2f%%' % (model.metrics_names[1],scores[1]*100))#in ra do chinh xac

# image=imresize(img.imread(link + 'test/' + test_dir_classes[ix] + '/' + test_dir_files[ix]), (299, 299))
# # predict_10_crop(np.array(image),0,debug=True,preprocess=False)
# print(test_dir_classes[ix])
# plt.imshow(img.imread(link + 'test/' + test_dir_classes[ix] + '/' + test_dir_files[ix]))
# plt.show()

# with urllib.request.urlopen('http://themodelhouse.tv/wp-content/uploads/2016/08/hummus.jpg') as f:
#     pic = plt.imread(f, format='jpg')
#     preds = predict_10_crop(np.array(pic), 0,preprocess=False)[0]
#     best_pred = collections.Counter(preds).most_common(1)[0][0]
#     print(ix_to_class[best_pred])
#     plt.imshow(pic)
#     plt.show()

def getImage(url,min_side=299):#resize image
    img_arr=np.array(img.imread(url))
    img_arr_rs=img_arr
    try:
        w = img_arr.shape[0]
        h = img_arr.shape[1]
        if w < min_side:
            wpercent = (min_side / float(w))
            hsize = int((float(h) * float(wpercent)))

            img_arr_rs = imresize(img_arr, (min_side, hsize))
        elif h < min_side:
            hpercent = (min_side / float(h))
            wsize = int((float(w) * float(hpercent)))

            img_arr_rs = imresize(img_arr, (wsize, min_side))
        return img_arr_rs
    except:
        print('Skipping bad image')

if os.path.exists('top_1.json'):
    with open('top_1.json','r') as f:
        preds_top_1=ast.literal_eval(f.read())
    with open('top_5.json','r') as f:
        preds_top_5=ast.literal_eval(f.read())
else:
    preds_10_crop = {}
    for ix in range(len(test_dir_files)):
        if ix % 1000 == 0:
            print(ix)
        value=getImage(link + 'test/' + test_dir_classes[ix] + '/' + test_dir_files[ix])
        if value is not None:
            preds_10_crop[ix] = predict_10_crop(value)

    preds_top_1 = {k: collections.Counter(v[0]).most_common(1) for k, v in preds_10_crop.items()}
    with open('top_1.json', 'w') as f:
        f.write(str(preds_top_1))

    top_5_per_ix = {k: collections.Counter(preds_10_crop[k][1].reshape(-1)).most_common(5)
                    for k, v in preds_10_crop.items()}
    preds_top_5 = {k: [y[0] for y in v] for k, v in top_5_per_ix.items()}
    with open('top_5.json', 'w') as f:
        f.write(str(preds_top_5))


right_counter = 0
for i in range(len(test_dir_classes)):
    guess, actual = preds_top_1[i][0][0], class_to_ix[test_dir_classes[i]]
    if guess == actual:
        right_counter += 1

print('Top-1 Accuracy, 10-Crop: {0:.2f}%'.format(right_counter / len(test_dir_files) * 100))

top_5_counter = 0
for i in range(len(test_dir_classes)):
    guesses, actual = preds_top_5[i], class_to_ix[test_dir_classes[i]]
    if actual in guesses:
        top_5_counter += 1

print('Top-5 Accuracy, 10-Crop: {0:.2f}%'.format(top_5_counter / len(test_dir_files) * 100))