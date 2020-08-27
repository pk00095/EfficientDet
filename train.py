"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import argparse
from datetime import date
import os
import sys
import tensorflow as tf

# import keras
# import keras.preprocessing.image
# import keras.backend as K
# from keras.optimizers import Adam, SGD

from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam, SGD

# from augmentor.color import VisualEffect
# from augmentor.misc import MiscEffect
from model import efficientdet, B0Config, B1Config, B2Config, B3Config, B4Config, B5Config, B6Config
from losses import smooth_l1, focal, smooth_l1_quad
from tfrecord_parser import Parser
# from efficientnet import BASE_WEIGHTS_PATH, WEIGHTS_HASHES


def makedirs(path):
    # Intended behavior: try to create the directory,
    # pass if the directory exists already, fails otherwise.
    # Meant for Python 2.7/3.n compatibility.
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


def create_callbacks(training_model, prediction_model, validation_generator, args):
    """
    Creates the callbacks to use during training.

    Args
        training_model: The model that is used for training.
        prediction_model: The model that should be used for validation.
        validation_generator: The generator for creating validation data.
        args: parseargs args object.

    Returns:
        A list of callbacks used for training.
    """
    callbacks = []

    tensorboard_callback = None

    if args.tensorboard_dir:
        if tf.version.VERSION > '2.0.0':
            file_writer = tf.summary.create_file_writer(args.tensorboard_dir)
            file_writer.set_as_default()
        tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir=args.tensorboard_dir,
            histogram_freq=0,
            batch_size=args.batch_size,
            write_graph=True,
            write_grads=False,
            write_images=False,
            embeddings_freq=0,
            embeddings_layer_names=None,
            embeddings_metadata=None
        )
        callbacks.append(tensorboard_callback)

    if args.evaluation and validation_generator:
        if args.dataset_type == 'coco':
            from eval.coco import Evaluate
            # use prediction model for evaluation
            evaluation = Evaluate(validation_generator, prediction_model, tensorboard=tensorboard_callback)
        else:
            from eval.pascal import Evaluate
            evaluation = Evaluate(validation_generator, prediction_model, tensorboard=tensorboard_callback)
        callbacks.append(evaluation)

    # save the model
    if args.snapshots:
        # ensure directory created first; otherwise h5py will error after epoch.
        makedirs(args.snapshot_path)
        checkpoint = keras.callbacks.ModelCheckpoint(
            os.path.join(
                args.snapshot_path,
                f'{args.dataset_type}_{{epoch:02d}}_{{loss:.4f}}_{{val_loss:.4f}}.h5' if args.compute_val_loss
                else f'{args.dataset_type}_{{epoch:02d}}_{{loss:.4f}}.h5'
            ),
            verbose=1,
            save_weights_only=True,
            # save_best_only=True,
            # monitor="mAP",
            # mode='max'
        )
        callbacks.append(checkpoint)

    # callbacks.append(keras.callbacks.ReduceLROnPlateau(
    #     monitor='loss',
    #     factor=0.1,
    #     patience=2,
    #     verbose=1,
    #     mode='auto',
    #     min_delta=0.0001,
    #     cooldown=0,
    #     min_lr=0
    # ))

    return callbacks






def main():

    from pprint import pprint

    config = B2Config()

    # pprint(vars(config))
    # exit()

    batch_size = 4
    num_classes = 15
    epochs = 5
    steps_per_epoch = 1000


    parser = Parser(
      config=config,
      batch_size=batch_size,
      num_classes=num_classes) 

    training_model = efficientdet(config, num_classes, weights=None)

    # compile model
    training_model.compile(
        optimizer=Adam(lr=1e-3), 
        loss={
            'regression': smooth_l1(),
            'classification': focal()})

    # print(training_model.summary())

    # # create the callbacks
    # callbacks = create_callbacks(
    #     model,
    #     prediction_model,
    #     validation_generator,
    #     args,
    # )


    train_dataset_function = parser.get_dataset(filenames='./DATA/train*.tfrecord')

    training_model.fit(
        train_dataset_function, 
        epochs=epochs, 
        steps_per_epoch=steps_per_epoch,)
        # callbacks=callbacks)

    os.makedirs("./checkpoints", exist_ok=True)

    training_model.save("./checkpoints/efficientdetB2_final")




if __name__ == '__main__':
    main()
