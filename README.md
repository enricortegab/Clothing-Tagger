# Clothing-Tagger

## Project Overview

**Etiquetador** is an AI-powered image tagging system designed to automate the labeling of clothing images. This allows users to perform intelligent natural language searches in an online store that continuously updates its product listings.

The system assigns two types of labels to images:
- **Color Labels** (unsupervised learning using K-means clustering)
- **Shape Labels** (supervised learning using K-Nearest Neighbors - KNN)

## Features

- **Automatic Color Tagging**: Detects the dominant colors in an image and assigns a color name based on universal color categories.
- **Automatic Shape Tagging**: Classifies images into predefined clothing categories using machine learning.
- **Image Search Engine**: Enables searches using color and shape labels (e.g., "red dress").
- **Optimized Performance**: Works with low-resolution images (60x80 pixels) to enhance processing speed.

## Dataset

The project uses the **Fashion Product Images Dataset** from Kaggle:
 [Dataset Link](https://www.kaggle.com/paramaggarwal/fashion-product-images-dataset)

## Technologies Used

- **Python**
- **Machine Learning Algorithms**: K-means clustering (for color detection) and KNN (for shape classification)
- **NumPy, OpenCV, Scikit-learn** (for image processing and ML)
- **Pandas** (for data handling)

## Project Structure

```
Etiquetador/
│── data/                     # Dataset and processed images
│── models/                   # Saved machine learning models
│── src/
│   │── kmeans.py             # K-means clustering for color tagging
│   │── knn.py                # KNN model for shape classification
│   │── utils.py              # Utility functions
│   │── my_labeling.py        # Search engine implementation
│── notebooks/                # Jupyter Notebooks for experimentation
│── README.md                 # Project documentation
```

## How It Works

1. **Color Tagging:** The system applies K-means clustering to detect the most dominant colors in an image and assigns labels from a predefined set of 11 universal colors.
2. **Shape Tagging:** The KNN classifier predicts the category of the clothing item (e.g., Shirt, Dress, Jeans).
3. **Search Engine:** Users can search for products using simple natural language queries, and the system retrieves matching items.

## Usage

To label an image, run:
```sh
python src/my_labeling.py --image path/to/image.jpg
```

For batch processing:
```sh
python src/my_labeling.py --batch path/to/dataset/
```

## Performance Analysis

- Evaluates different K values in K-means for optimal color classification.
- Tests various feature extraction techniques for shape classification.
- Performance metrics: Accuracy, F1-score, and execution time.


## Contact

For any inquiries, please contact **enricortegab@gmail.com* or open an issue in the repository.

