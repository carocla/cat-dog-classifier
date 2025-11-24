# Adapted from Géron

import tensorflow as tf
from tensorflow.keras import layers
from keras.callbacks import EarlyStopping
import kagglehub, pathlib


# 1. Download dataset, point to PetImages folder
path = kagglehub.dataset_download("karakaggle/kaggle-cat-vs-dog-dataset")
print("Database root:", path)
data_dir = pathlib.Path(path) / "kagglecatsanddogs_3367a" / "PetImages"
print("Using data_dir:", data_dir)

IMG_SIZE = 64
BATCH_SIZE = 32

# Build raw datasets
train_ds_raw = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=42, #random
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)
val_test_ds_raw = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=42, #random seed
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

# Determine batch size, move 2/3 of them to test set
val_batches = tf.data.experimental.cardinality(val_test_ds_raw)
test_ds_raw = val_test_ds_raw.take((2*val_batches) // 3)
val_ds_raw = val_test_ds_raw.skip((2*val_batches) // 3)

# Prefetch prepares next batch of data in background
# Auto-determines  # of batches to prefetch based on sys resources
train_ds = (train_ds_raw.ignore_errors()
            .shuffle(256)
            .prefetch(tf.data.AUTOTUNE))
val_ds = (val_ds_raw.ignore_errors()
          .prefetch(tf.data.AUTOTUNE))
test_ds = (test_ds_raw.ignore_errors()
          .prefetch(tf.data.AUTOTUNE))

# Define the model
model = tf.keras.Sequential([
    # Normalize images to [0,1] range
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.Rescaling(1. / 255),
    tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation="relu",
                           padding = "same", kernel_initializer="he_normal",
                           input_shape=[28, 28, 1]),
    tf.keras.layers.MaxPooling2D(pool_size=2),
    tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation="relu",
                           padding = "same", kernel_initializer="he_normal"),
    layers.MaxPooling2D(pool_size=2),
    tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation="relu",
                           padding="same", kernel_initializer="he_normal"),
    tf.keras.layers.MaxPooling2D(pool_size=2),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.5),
    # Each img has 1 output value: its binary classification
    layers.Dense(1, activation="sigmoid")
])

# Compile the model
model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

early_stopping = EarlyStopping(monitor="val_loss",
                               patience=3,
                               restore_best_weights=True)
# Train the model
history = model.fit(
    train_ds,
    epochs=30,
    validation_data=val_ds,
    callbacks = [early_stopping])

# Evaluate the model on test data
model.evaluate(test_ds)
