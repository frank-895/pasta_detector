# Pasta Detector 🍝

A deep learning model that can identify different pasta shapes based on images. Trained using fastai and deployed on Hugging Face Spaces. 

Visit the Hugging Face Space to interact with the model:

[Pasta Detector on Hugging Face Spaces](https://huggingface.co/spaces/frank-895/pasta_detector)

## Table of Contents

1. [Overview](#overview)
2. [Project Setup](#project-setup)
3. [Model Training](#model-training)
4. [Web Application](#web-application)
5. [How to Use](#how-to-use)
6. [Improvements](#improvements)

## Overview

This project uses a deep learning model to classify pasta shapes from an image. The model was trained on images of pasta shapes, scraped using DuckDuckGo, and fine-tuned using the fastai library. It can classify the following pasta types:

- Spaghetti
- Penne
- Fusilli
- Farfalle
- Fettuccine
- Macaroni
- Orecchiette
- Gnocchi

The trained model was then exported and integrated into a Gradio web application, hosted on Hugging Face Spaces, where users can upload pasta images to predict the pasta type.

## Project Setup

To set up this project locally, ensure that you have the following dependencies installed:

```bash
pip install fastbook fastai duckduckgo-search gradio
```

## Dataset Creation
The dataset was created by searching for images of different pasta shapes using DuckDuckGo and downloading them to a local directory. Any failed or unusable images were discarded to ensure the quality of the dataset.

## Model Training
### Data Preprocessing
I used fastai's DataBlock API to create a data pipeline for loading and augmenting images. The images were resized and split into training and validation sets.
```
dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=RandomResizedCrop(224, min_scale=0.3),
    batch_tfms=aug_transforms()
).dataloaders(path)
```
### Model Creation and Fine-tuning
I used a pretrained ResNet18 model for transfer learning, fine-tuning it on the pasta dataset.
```
learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(2)
```
### Model Evaluation
We evaluated the model using a confusion matrix and improved the model by cleaning up misclassified images using ```ImageClassifierCleaner```.

After addressing errors by adding more images for certain pasta types, we exported the model.

## Web Application
The trained model was integrated into a Gradio web application that classifies pasta images. The app is hosted on Hugging Face Spaces, and users can upload images to get pasta predictions.

### How to Use
Visit the Hugging Face Space to interact with the model:

[Pasta Detector on Hugging Face Spaces](https://huggingface.co/spaces/frank-895/pasta_detector)

Upload an image of pasta, and the model will classify it, providing the top 3 most likely pasta types along with the probabilities.

## Improvements
The trained model achieved an error rate of 14% on the test set, indicating that it correctly classified pasta shapes with high accuracy. However, some pasta shapes, such as farfalle and macaroni, were occasionally confused with others, particularly penne. Further improvements can be made by:
- Increase Dataset Size: We used less than 100 images for most categories. Increasing the dataset can improve accuracy.
- Improve Data Quality: Manually curate images for better quality and consistency.
- Use More Complex Models: Experiment with deeper architectures like ResNet34 or ResNet50.
- Training for More Epochs: The model was trained for only 2 epochs, so training for more epochs could improve results.

This project is licensed under the Apache 2.0 License.
