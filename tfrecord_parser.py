"""Summary
"""
import tensorflow as tf
from tensorflow import keras
import cv2, os, glob
import numpy as np
# from .preprocessing import anchor_targets_bbox, anchors_for_shape

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


def rescale_bboxes(bboxes, scale):

  batch_size, num_boxes, num_annotations = bboxes.shape

  assert scale.ndim == 2

  # scale = np.expand_dims(scale,-1)
  scale_matrix = np.tile(scale, num_boxes*num_annotations).reshape(bboxes.shape)

  return bboxes*scale_matrix

class Parser(object):
  """docstring for Parser"""
  def __init__(self, batch_size, num_classes, image_size):
    # super(Parser, self).__init__()
    self.batch_size = batch_size
    self.image_size = image_size
    self.num_classes = num_classes
    
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
          'image/object/bbox/xmin': tf.io.VarLenFeature(tf.keras.backend.floatx()),
          'image/object/bbox/xmax': tf.io.VarLenFeature(tf.keras.backend.floatx()),
          'image/object/bbox/ymin': tf.io.VarLenFeature(tf.keras.backend.floatx()),
          'image/object/bbox/ymax': tf.io.VarLenFeature(tf.keras.backend.floatx()),
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

        bboxes_batch = tf.concat([xmin_batch, xmax_batch, ymin_batch, ymax_batch], axis=-1)

        bboxes_batch = tf.numpy_function(rescale_bboxes,(bboxes_batch, scale_batch), Tout=keras.backend.floatx())

        return image_batch, {'bboxes':bboxes_batch, 'labels':label_batch}


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

    from helpers import draw_boxes_on_image

    batch_size = 4

    parser = Parser(
      batch_size=batch_size,
      num_classes=5,
      image_size=300)

    dataset = parser.get_dataset(filenames='./DATA/train*.tfrecord')

    i = 0

    for x,y in dataset.take(4):
        image_batch = x.numpy()

        bboxes = y['bboxes'].numpy()
        labels = y['labels'].numpy()

        print(image_batch.shape, bboxes.shape, labels.shape)
        print(image_batch.dtype, bboxes.dtype, labels.dtype)

        for index in range(batch_size):
          annotated_arr = draw_boxes_on_image(image_batch[index], bboxes[index], labels[index])

          cv2.imwrite(f"{i}.jpg", annotated_arr)

          i+= 1

        # print(scales.shape)
        # print(scales)


