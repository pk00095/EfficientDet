import tensorflow as tf
from tensorflow import keras
import os
import numpy as np

from utils.anchors import anchors_for_shape
from model import BoxNet, ClassNet, B0Config
from layers import RegressBoxes, ClipBoxes, FilterDetections

from tfrecord_parser import Parser
from helpers import annotate_image as image_annotator
    # from helpers import draw_boxes_on_image, B0Config

write_out_dir = './results'

os.makedirs(write_out_dir, exist_ok=True)

def get_dummy_model(config, score_threshold=0.5):
    regression = keras.Input(shape=(49104, 5))
    classification = keras.Input(shape=(49104, 16))

    anchors = anchors_for_shape(
        image_shape=config.input_shape,
        sizes=config.sizes,
        ratios=config.ratios,
        scales=config.scales,
        strides=config.strides,
        pyramid_levels=[3, 4, 5, 6, 7])
    anchors = tf.convert_to_tensor(anchors)
    anchors_input = tf.expand_dims(anchors, axis=0)
    boxes = RegressBoxes(name='boxes')([anchors_input, regression[..., :4]])
    # boxes = ClipBoxes(name='clipped_boxes')([model.input, boxes])

    # filter detections (apply NMS / score threshold / select top-k)

    detections = FilterDetections(
        name='filtered_detections',
        score_threshold=score_threshold
    )([boxes, classification])

    prediction_model = keras.models.Model(inputs=[regression, classification], outputs=detections, name='efficientdet_bo')
    return prediction_model


# if __name__ == '__main__':
    # filepath = os.path.join(os.getcwd(),'DATA','train*.tfrecord')


config = B0Config()

batch_size = 4

parser = Parser(
  config=config,
  batch_size=batch_size,
  num_classes=15,
  training=False)

dataset = parser.get_dataset(filenames='./DATA/train*.tfrecord')

dummy_model = get_dummy_model(config)

i = 0
confidence_threshold = 0.75

for x,y in dataset.take(4):
    image_batch = x.numpy()

    input_bboxes = y['regression'].numpy()
    input_labels = y['classification'].numpy()

    print(image_batch.shape, input_bboxes.shape, input_labels.shape)
    print(image_batch.dtype, input_bboxes.dtype, input_labels.dtype)

    # print(np.unique(bboxes[:,:,-1]))
    # print(np.unique(labels[:,:,-1]))

    bboxes, scores, labels = dummy_model.predict([input_bboxes, input_labels])

    print(bboxes.shape, scores.shape, labels.dtype)

    for index in range(batch_size):

        bbox = bboxes[index]
        confidence = scores[index]
        label = labels[index]

        # print(bbox.shape)
        # confidence

        # print(np.unique(label))

        annotated_image = image_annotator(
            image=image_batch[index], 
            bboxes=bbox.astype(int), 
            scores=confidence, 
            labels=label, 
            threshold=0.75)

        i += 1


        annotated_image.save(os.path.join(write_out_dir,f"{i}_predicted.jpg"))

