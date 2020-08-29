import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from model import efficientdet, B0Config, B1Config, B2Config, B3Config, B4Config, B5Config, B6Config
from model import freeze_model


import os, glob
import numpy as np

from tensorflow.keras.applications.imagenet_utils import preprocess_input
from helpers import annotate_image as image_annotator

from model import ClassNet, BoxNet
from initializers import PriorProbability


config = B0Config()
num_classes = 15

keras.backend.clear_session()

training_model = efficientdet(config, num_classes, weights=None)
training_model.load_weights('./checkpoints/efficientdetB0_final.h5')

# training_model = keras.models.load_model('./checkpoints/efficientdetB0_final.h5', 
#     compile=False, 
#     # custom_objects={'ClassNet':ClassNet, 'BoxNet':BoxNet, 'PriorProbability':PriorProbability})
#     custom_objects={'PriorProbability':PriorProbability})

prediction_model = freeze_model(model=training_model, config=config)


orig_images = [] # Store the images here.
input_images = []
pil_input_images = [] # Store resized versions of the images here.

# We'll only load one image in this example.
img_dir = './example_images'
write_out_dir = './results'

os.makedirs(write_out_dir, exist_ok=True)

for image_path in glob.glob(os.path.join(img_dir,'*.jpg')):
    pil_image = image.load_img(image_path, target_size=(config.height, config.width))
    pil_input_images.append(pil_image)
    img = np.array(pil_image)
    input_images.append(img)

input_images = preprocess_input(np.array(input_images))

bboxes, scores, labels = prediction_model.predict(input_images)


confidence_threshold = 0.75


for index in range(input_images.shape[0]):

    bbox = bboxes[index]
    confidence = scores[index]
    label = labels[index]

    print(bbox.shape)
    # confidence

    annotated_image = image_annotator(
        image=pil_input_images[index], 
        bboxes=bbox.astype(int), 
        scores=confidence, 
        labels=label, 
        threshold=0.75)


    annotated_image.save(os.path.join(write_out_dir,f"{index}_predicted.jpg"))