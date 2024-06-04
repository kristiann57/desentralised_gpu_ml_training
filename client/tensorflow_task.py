import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import os
import shutil


# ------ ENSURE DIRECTRY WHERE SAVED MODEL SAVED EXISTS ------- #####


def ensure_directory():
    model_dir = 'temp'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)  # This creates the directory if it does not exist
    return model_dir



def load_data():
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)
    encoder = OneHotEncoder(sparse=False)
    y_train = encoder.fit_transform(y_train[:, np.newaxis])
    y_test = encoder.transform(y_test[:, np.newaxis])
    return X_train, X_test, y_train, y_test

def build_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(10, activation='relu', input_shape=(4,)),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# ------ TRAIN THE MODEL AND SAVE IT ------- #####
# returns the path to the saved model zip file
def tensorflow_train_model():
    model_dir = ensure_directory()
    X_train, X_test, y_train, y_test = load_data()
    model = build_model()
    model.fit(X_train, y_train, epochs=50, verbose=0)
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    model_path = os.path.join(model_dir, 'saved_model')
    tf.saved_model.save(model, model_path)
    
    # Compress the saved model directory
    shutil.make_archive(model_path, 'zip', model_path)
    
    return f"{model_path}.zip"

if __name__ == '__main__':
    model_zip_path = tensorflow_train_model()
    print(model_zip_path)
