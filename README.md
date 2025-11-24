# Cat vs Dog Image Classifier 

A convolutional neural network (CNN) that classifies images as either **cat** or **dog**, trained on the [Kaggle Cats vs Dogs dataset](https://www.kaggle.com/datasets/karakaggle/kaggle-cat-vs-dog-dataset).

## Tech Stack

- Python
- TensorFlow / Keras

## What This Repo Includes

- Data loading with `tf.keras.utils.image_dataset_from_directory`
- CNN model definition (Conv2D + MaxPooling + Dense layers)
- Training with early stopping
- Evaluation and plotting of accuracy and loss

## Data & Preprocessing

- Images are loaded from the Kaggle dataset, resized to **64×64**, and normalized.
- The dataset contains a small number of corrupt / non-standard images (e.g., unexpected channel counts), so the `tf.data` pipeline uses `Dataset.ignore_errors()` to **skip any samples that fail to decode**, keeping training robust without manual dataset cleanup.
