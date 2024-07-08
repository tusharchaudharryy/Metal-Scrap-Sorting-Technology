# Metal Scrap Sorting

This project uses a Convolutional Neural Network (CNN) to classify images of metal scrap into different categories, including Aluminium Scrap, Copper Scrap, Stainless Steel Scrap, and Steel Scrap.

## Installation

1. Clone this repository:
```bash
git clone https://github.com/your-username/metal-scrap-sorting.git
cd metal-scrap-sorting
```
2. Install the required packages:
```bash
pip install -r requirements.txt
```
## Dataset Preparation

Organize your dataset into subdirectories for each class under a directory named data.
Ensure the directory structure follows:
```bash
/data
    ├── Aluminium
    ├── Copper
    ├── StainlessSteel
    └── Steel
```
## Training the Model

- Run the Jupyter notebook or Python script provided to preprocess the dataset, train the model, and visualize training results.
- Evaluation and Prediction
- Evaluate the model using the test set and visualize accuracy and loss.
- Use the trained model to predict the class of new images.

## Example Usage

python
```bash
import cv2
import numpy as np
import tensorflow as tf

img = cv2.imread('/path/to/your/image.jpg')
resize = tf.image.resize(img, (256,256))
yhat = model.predict(np.expand_dims(resize/255, 0))

class_idx = np.argmax(yhat)
classes = ["Aluminium Scrap", "Copper Scrap", "Stainless Steel Scrap", "Steel Scrap"]
print(classes[class_idx])
```
