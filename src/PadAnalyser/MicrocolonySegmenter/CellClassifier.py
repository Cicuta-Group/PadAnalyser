import numpy as np
import os, glob, shutil
import seaborn as sns

import MKUtils
from MKImageAnalysis import PlotPads, MKAnalysisUtils, MODEL_VERSION, SEGMENTATION_VERSION, experiment_folder_name

import pathlib

import matplotlib.pyplot as plt
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # to supress tensorlfow warnings
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

''' 
To train model:

Workflow for a given experiment: 

- Copy relevant colony masks to directory
- Apply classificaiotn algoritm and copy images to prediction folders
- Add column to dataframe based on predictions
'''

MODEL_NAME = f'model_{MODEL_VERSION}.tflite'


def save_model(model, model_path):
    # Convert the model.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save the model
    with open(model_path, 'wb') as f:
        f.write(tflite_model)


def load_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    l = interpreter.get_signature_list()['serving_default']
    return interpreter.get_signature_runner('serving_default'), l['inputs'][0], l['outputs'][0]


def copy_relevant_colony_masks(df, input_path, output_path):
    MKUtils.generate_directory(output_path)
    print(input_path)
    filenames = os.listdir(input_path)
    label_names = [f.split('_x')[0] for f in filenames]

    for value, df_query in df.groupby(PlotPads.SERIES_KEY):
        label = df_query['label'].iloc[0]
        name = df_query['colony_name'].iloc[0]
        label_name = f'{label}_n{name}'
        i = label_names.index(label_name)
        shutil.copy(os.path.join(input_path, filenames[i]), output_path)


def add_is_debris_ml_column(df, cell_path):
    files = os.listdir(cell_path)
    label_names = [f.split('_x')[0] for f in files]

    for _, df_query in df.groupby(PlotPads.SERIES_KEY):
        label = df_query['label'].iloc[0]
        name = df_query['colony_name'].iloc[0]
        label_name = f'{label}_n{name}'
        df.loc[df_query.index, 'is_debris_ml'] = label_name not in label_names


# returns prediciton score from 0 to 1 of mask beeing a cell based on current model
def mask_is_cell_score(filename, model, model_input, model_output):
    img = tf.keras.utils.load_img(filename, target_size=(img_height, img_width))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = eval(f'model({model_input}=img_array)[model_output]')
    score = tf.nn.softmax(predictions)
    return float(score[0][0])

# Takes the path to all masks, and adds column to dataframe with confidence of that colony beeing debris
# Does not move any images -> much faster, but harder to verify
def add_is_debris_score_column(df):
    mask_directory = config('WORK_DIRECTORY')
    model_path = os.path.join(mask_directory, MODEL_NAME)
    model, model_input, model_output = load_model(model_path)

    for experiment, df_g in df.groupby('experiment'):
        all_masks_path = os.path.join(mask_directory, experiment_folder_name(experiment), 'all_masks')
        files = os.listdir(all_masks_path)
        label_names = [f.split('_x')[0] for f in files]

        for _, df_query in df_g.groupby(PlotPads.SERIES_KEY):
            label = df_query['label'].iloc[0]
            name = df_query['colony_name'].iloc[0]
            label_name = f'{label}_n{name}'

            try:
                i = label_names.index(label_name)
                is_cell_score = mask_is_cell_score(filename=os.path.join(all_masks_path, files[i]), model=model, model_input=model_input, model_output=model_output)
            except Exception as e:
                logging.error(f'Could not find the mask {label_name} in {all_masks_path}. {e}')
                is_cell_score = np.NaN

            df.loc[df_query.index, 'is_cell_score'] = is_cell_score




batch_size = 32
img_height = 128
img_width = 128

def generate_model(data_dir):
    data_dir = pathlib.Path(data_dir)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size,
    )

    class_names = train_ds.class_names
    print(f'{class_names=}')

    # Dataset configuration
    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # standardising data
    normalization_layer = layers.Rescaling(1./255)

    normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    image_batch, labels_batch = next(iter(normalized_ds))
    first_image = image_batch[0]
    # Notice the pixel values are now in `[0,1]`.
    print(np.min(first_image), np.max(first_image))

    # generate model

    num_classes = len(class_names)

    model = Sequential([
        layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes),
    ])

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    # print(model.summary())

    epochs=10
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
    )

    return model, history, epochs, class_names


def predict_all_to_folder(source_path, prediction_path, model_filename):

    model, model_input, model_output = load_model(model_filename)
    
    # assert class_names == ['cells', 'debris'] # if not, may need to update some stuff!

    cell_path = os.path.join(prediction_path, f'cells')
    debris_path = os.path.join(prediction_path, f'debris')

    MKUtils.generate_directory(cell_path, clear=True)
    MKUtils.generate_directory(debris_path, clear=True)

    test_files = glob.glob(f'{source_path}/*.jpg')

    for f in test_files:
        
        cell_score = mask_is_cell_score(f, model, model_input, model_output)
        
        # used for training
        out_path = cell_path if cell_score > 0.5 else debris_path
        shutil.copy(f, out_path) 
        

import logging
import pandas as pd

def generate_predictions_for_experiment(df: pd.DataFrame, mask_directory: str, output_mask_directory: str):
    
    for experiment, df_g in df.groupby('experiment'):
        logging.info(f'Debris model for {experiment}')
        
        input_directory = os.path.join(mask_directory, experiment_folder_name(experiment), 'all_masks')
        output_experiment_directory = os.path.join(output_mask_directory, experiment_folder_name(experiment))
        
        MKUtils.generate_directory(output_experiment_directory)

        relevant_masks_directory = os.path.join(output_experiment_directory, 'relevant_masks')
        predictions_path = os.path.join(output_experiment_directory, 'predictions')

        model_path = os.path.join(output_mask_directory, MODEL_NAME)
        print(model_path, output_mask_directory)

        logging.info(f'Copy relevant colonies from {input_directory} to {relevant_masks_directory}')
        copy_relevant_colony_masks(df_g, input_path=input_directory, output_path=relevant_masks_directory)
        
        logging.info(f'Predict on masks in {relevant_masks_directory} to {predictions_path} using model {model_path}')
        predict_all_to_folder(source_path=relevant_masks_directory, prediction_path=predictions_path, model_filename=model_path)

        logging.info(f'Add column to dataframe from {CONFIDENT_NAME}')
        add_is_debris_ml_column(df=df_g, cell_path=os.path.join(predictions_path, CONFIDENT_NAME))