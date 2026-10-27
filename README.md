# Brain Tumor AI

Brain MRI tumor classification with TensorFlow/Keras, using a pre-trained
InceptionV3 model (GoogLeNet family).

Classes: glioma, meningioma, notumor, pituitary.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Data

Dataset: https://www.kaggle.com/datasets/ahmedsorour1/mri-for-brain-tumor-with-bounding-boxes

Put the images in one folder per class:

```
data/classification/raw/glioma/
data/classification/raw/meningioma/
data/classification/raw/notumor/
data/classification/raw/pituitary/
```

## Notebooks

Run them in order from the `notebooks/` folder:

1. `01_data_exploration.ipynb` - check the images, count per class, show samples
2. `02_data_preparation.ipynb` - split 70/15/15 into `data/classification/processed/`, augment the train set (blur, noise, flip)
3. `03_classification_training.ipynb` - train the model and save it to `models/classification/`
