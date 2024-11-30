# Tree Detection from Satellite Imagery

This repository contains a complete pipeline for detecting trees in satellite imagery using machine learning. The system includes tools for data collection, annotation, model training, and prediction.

## Repository Structure

### Data Collection and Annotation
- `Tool.py` - Interactive tool for annotating trees in satellite images
- `ImageCollection.py` - Utility for fetching satellite imagery from Mapbox
- `/annotations` - Contains annotated tree locations
- `/images` - Satellite imagery from the Pacific Northwest
- `/masks` - Generated masks for training
- `/incorrect_points` - Logged incorrect predictions for model improvement

### Model Training
- `trainingtree.py` - Main training script for the tree classification model
- `TreeDataset.py` - Dataset preparation and processing utilities
- `simple_tree_classifier.h5` - Trained model file
- `tree_cnn_classifier.h5` - CNN-based model file
- `tree_classifier.joblib` - Alternative classifier model

### Prediction
- `treeprediction.py` - Script for running predictions on new satellite images

### Additional Resources
- `/not_tree_annotations` - Negative samples for training
- `/not_tree_incorrect_points` - Logged false positives
- `/not_tree_masks` - Masks for non-tree areas

## Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up Mapbox credentials for image collection:
   ```python
   # config.py
   MAPBOX_TOKEN = 'your_token_here'
   ```

3. Collect satellite imagery:
   ```bash
   python ImageCollection.py
   ```

4. Run the annotation tool:
   ```bash
   python Tool.py
   ```

5. Train the model:
   ```bash
   python trainingtree.py
   ```

6. Make predictions:
   ```bash
   python treeprediction.py
   ```

## Features

- **Interactive Annotation Tool**: Quickly create training datasets by marking trees in satellite images
- **Automated Data Collection**: Integration with Mapbox for satellite imagery
- **Flexible Training Pipeline**: Support for different model architectures
- **Prediction Optimization**: Includes preprocessing to skip unlikely tree locations
- **Error Logging**: Tracks incorrect predictions for model improvement

## Model Training

The system uses a CNN-based approach to classify regions as trees or non-trees. The training process includes:
1. Data augmentation for improved generalization
2. Balanced sampling of positive and negative examples
3. Validation on a held-out test set
4. Error analysis and model refinement

## Contributing

We welcome contributions! Please read `CONTRIBUTING.md` for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Acknowledgments

- MapBox for satellite imagery
- Pacific Northwest dataset contributors
- [Add any other acknowledgments]