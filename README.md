Image Captioning with TensorFlow on COCO Dataset

## Synopsis

This repository contains the code for an image captioning project implemented in TensorFlow. The model is trained on the COCO (Common Objects in Context) 2017 dataset to generate descriptive captions for images. This project serves as a practical example of how to build and train a deep learning model for a creative NLP task.

## Dataset

The project utilizes the **COCO 2017 dataset**, a large-scale object detection, segmentation, and captioning dataset. The code is configured to work with the caption annotations and the corresponding images from this dataset.

How it Works

The core of the project is a neural network model that combines computer vision and natural language processing techniques. Here's a high-level overview of the process:

1.  **Image Feature Extraction**: A pre-trained convolutional neural network (CNN) is used to extract feature vectors from the input images.
2.  **Caption Preprocessing**: The text captions are preprocessed by:
      * Converting all text to lowercase.
      * Removing punctuation and special characters.
      * Adding `[start]` and `[end]` tokens to signify the beginning and end of each caption.
3.  **Model Training**: The image features and preprocessed captions are used to train a recurrent neural network (RNN), likely an LSTM or GRU, which learns to generate captions word by word based on the image content.
4.  **Inference**: Once trained, the model can take a new image as input and generate a caption for it.

##  Getting Started

### Prerequisites

Make sure you have the following libraries installed:

  * TensorFlow
  * pandas
  * NumPy
  * Matplotlib
  * Pillow (PIL)
  * tqdm

You can install them using pip:

```bash
pip install tensorflow pandas numpy matplotlib pillow tqdm
```

### Usage

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/SaiChetanM51/Image-Captioning-with-COCO-DATASET.git
    cd Image-Captioning-with-COCO-DATASET
    ```
2.  **Download the dataset**:
    Download the COCO 2017 dataset and place it in the appropriate directory as referenced by the `BASE_PATH` variable in the notebook.
3.  Install the required dependencies:
  ```bash
     pip install -r requirements.txt
  ```
5.  **Run the Jupyter Notebook**:
    Open and run the `image-captioning-on-coco-dataset (1).ipynb` notebook to train the model and generate captions.
6.  **Load pre-trained weights**:
    If you have pre-trained weights (`model.h5`), you can load them to an instantiated model to perform inference without retraining.

-----
