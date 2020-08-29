"""Summary
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import cv2, os, glob
import numpy as np
# from preprocessing import anchor_targets_bbox, anchors_for_shape
from utils.anchors import anchors_for_shape, anchor_targets_bbox

def rescale_image(image, image_size):
    image_height, image_width = image.shape[:2]
    if image_height > image_width:
        scale = image_size / image_height
        resized_height = image_size
        resized_width = int(image_width * scale)
    else:
        scale = image_size / image_width
        resized_height = int(image_height * scale)
        resized_width = image_size

    image = cv2.resize(image, (resized_width, resized_height))

    pad_h = image_size - resized_height
    pad_w = image_size - resized_width
    image = np.pad(image, [(0, pad_h), (0, pad_w), (0, 0)], mode='constant')

    return image, keras.backend.cast_to_floatx(scale)

@tf.function
def decode_pad_resize(image_string, image_size):
  """Summary
  
  Args:
      image_string (TYPE): Description
      pad_height (TYPE): Description
      pad_width (TYPE): Description
      scale (TYPE): Description
  
  Returns:
      tf.tensor: Description
  """
  image = tf.image.decode_jpeg(image_string)
  image, scale = tf.numpy_function(rescale_image, [image, image_size], Tout=[tf.uint8, keras.backend.floatx()])
  #image.set_shape([None, None, 3])
  return image #, scale

def rescale_bboxes(bboxes, scale):

  batch_size, num_boxes, num_annotations = bboxes.shape

  assert scale.ndim == 2

  # scale = np.expand_dims(scale,-1)
  scale_matrix = np.tile(scale, num_boxes*num_annotations).reshape(bboxes.shape)

  return bboxes*scale_matrix

class Parser(object):
  """docstring for Parser"""
  def __init__(self, config, batch_size, num_classes, training=True):
    # super(Parser, self).__init__()
    self.batch_size = batch_size
    self.image_size = config.height
    self.num_classes = num_classes
    self.config = config
    self._is_training = training


  def process_bboxes(self, image_array, bboxes, labels):

      # delete bboxes containing [-1,-1,-1,-1]
      bboxes = bboxes[~np.all(bboxes<=-1, axis=1)]
      # delete labels containing[-1]
      labels = labels[labels>-1]#[0]

      # generate raw anchors
      raw_anchors = anchors_for_shape(
          image_shape=image_array.shape,
          sizes=self.config.sizes,
          ratios=self.config.ratios,
          scales=self.config.scales,
          strides=self.config.strides,
          pyramid_levels=[3, 4, 5, 6, 7],
          shapes_callback=None,
      )

      # generate anchorboxes and class labels      
      gt_regression, gt_classification = anchor_targets_bbox(
            anchors=raw_anchors,
            image=image_array,
            bboxes=bboxes,
            labels=labels,
            num_classes=self.num_classes,
            negative_overlap=0.4,
            positive_overlap=0.5
        )

      return gt_regression, gt_classification

  @tf.function
  def tf_process_bboxes(self, image_batch, bboxes_batch, label_batch):

      regression_batch = list()
      classification_batch = list()

      for index in range(self.batch_size):
          bboxes, labels = bboxes_batch[index], label_batch[index]
          image_array = image_batch[index]
          # bboxes = tf.convert_to_tensor([xmins,ymins,xmaxs,ymaxs], dtype=keras.backend.floatx())
          # bboxes = tf.transpose(bboxes)
          gt_regression, gt_classification = tf.numpy_function(self.process_bboxes, [image_array, bboxes, labels], Tout=[keras.backend.floatx(), keras.backend.floatx()])

          regression_batch.append(gt_regression)
          classification_batch.append(gt_classification)

      return tf.convert_to_tensor(regression_batch), tf.convert_to_tensor(classification_batch)
    
  def _parse_function(self,serialized):
        """Summary
        
        Args:
            serialized (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        features = {
          'image/height': tf.io.FixedLenFeature([], tf.int64),
          'image/width': tf.io.FixedLenFeature([], tf.int64),
          'image/encoded': tf.io.FixedLenFeature([],tf.string),
          'image/object/bbox/xmin': tf.io.VarLenFeature(keras.backend.floatx()),
          'image/object/bbox/xmax': tf.io.VarLenFeature(keras.backend.floatx()),
          'image/object/bbox/ymin': tf.io.VarLenFeature(keras.backend.floatx()),
          'image/object/bbox/ymax': tf.io.VarLenFeature(keras.backend.floatx()),
          'image/f_id': tf.io.FixedLenFeature([], tf.int64),
          'image/object/class/label':tf.io.VarLenFeature(tf.int64)}


        parsed_example = tf.io.parse_example(serialized=serialized, features=features)

        height_batch = parsed_example['image/height']
        width_batch = parsed_example['image/width']

        max_dims = tf.maximum(height_batch, width_batch)

        scale_vector = keras.backend.cast_to_floatx(self.image_size/max_dims)

        scale_batch = tf.expand_dims(scale_vector, axis=-1)

        image_batch = tf.map_fn(lambda x: decode_pad_resize(x, self.image_size), parsed_example['image/encoded'], dtype=tf.uint8)

        xmin_batch = tf.expand_dims(tf.sparse.to_dense(parsed_example['image/object/bbox/xmin'], default_value=-1), axis=-1) #*scale_matrix
        xmax_batch = tf.expand_dims(tf.sparse.to_dense(parsed_example['image/object/bbox/xmax'], default_value=-1), axis=-1) #*scale_matrix
        ymin_batch = tf.expand_dims(tf.sparse.to_dense(parsed_example['image/object/bbox/ymin'], default_value=-1), axis=-1) #*scale_matrix
        ymax_batch = tf.expand_dims(tf.sparse.to_dense(parsed_example['image/object/bbox/ymax'], default_value=-1), axis=-1) #*scale_matrix

        label_batch = tf.expand_dims(tf.sparse.to_dense(parsed_example['image/object/class/label'], default_value=-1), axis=-1)
        # label_batch = tf.sparse.to_dense(parsed_example['image/object/class/label'])

        bboxes_batch = tf.concat([xmin_batch, ymin_batch, xmax_batch, ymax_batch], axis=-1)

        bboxes_batch = tf.numpy_function(rescale_bboxes,(bboxes_batch, scale_batch), Tout=keras.backend.floatx())

        regression_batch, classification_batch = self.tf_process_bboxes(image_batch, bboxes_batch, label_batch) #], Tout=[keras.backend.floatx(), keras.backend.floatx()])

        if self._is_training:
          image_batch = preprocess_input(image_batch)

        return image_batch , {'regression':regression_batch, 'classification':classification_batch}


  def get_dataset(self, filenames):
        # dataset = tf.data.Dataset.from_tensor_slices(filenames).repeat(-1)
        dataset = tf.data.Dataset.list_files(filenames).shuffle(buffer_size=8).repeat(-1)

        dataset = dataset.interleave(
          tf.data.TFRecordDataset, 
          num_parallel_calls=tf.data.experimental.AUTOTUNE,
          deterministic=False)

        dataset = dataset.batch(
          self.batch_size, 
          drop_remainder=True)    # Batch Size

        dataset = dataset.map(
          self._parse_function, 
          num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # dataset = dataset.cache()
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return dataset


if __name__ == '__main__':
    # filepath = os.path.join(os.getcwd(),'DATA','train*.tfrecord')

    from helpers import draw_boxes_on_image, B0Config

    config = B0Config()

    batch_size = 4

    parser = Parser(
      config=config,
      batch_size=batch_size,
      num_classes=15)

    dataset = parser.get_dataset(filenames='./DATA/train*.tfrecord')

    i = 0

    import pdb

    for x,y in dataset.take(4):
        image_batch = x.numpy()

        bboxes = y['regression'].numpy()
        labels = y['classification'].numpy()

        for index in range(batch_size):

          print("==================================")
          print(f'images shape {image_batch[index].shape}')
          print(f'bboxes shape {bboxes[index].shape}')
          print(f'labels shape {labels[index].shape}')
          print(labels.dtype)

          # print(f'unique labels {np.unique(np.argmax(labels[index,:,:-1], axis=-1))}')

          ignore_index_classification = labels[index,:,-1]
          ignore_index_regression = bboxes[index,:,-1]

          print(f'ignore index equality of regression and classification == {np.array_equal(ignore_index_classification, ignore_index_regression)}')

          current_labels = labels[index]

          ignore_index = current_labels[current_labels[:,-1]==-1]
          positive_index = current_labels[current_labels[:,-1]==1]
          negative_index = current_labels[current_labels[:,-1]==0]

          print(f'ignore index = {np.unique(np.argmax(ignore_index, axis=-1))}')
          print(f'positive_index = {np.unique(np.argmax(positive_index, axis=-1))}')
          print(f'negative_index = {np.unique(np.argmax(negative_index, axis=-1))}')


