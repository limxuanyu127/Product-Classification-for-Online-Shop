# Product Classification for Online Shop

## Description
Machine-learning models to identify the tag(eg. dress, eyeliner, book etc.) of any item, given a picture and short text description of it.
This is part of a submission for the Shopee National Data Science Challenge 2018.

## Code Details
The project focuses on 3 different methodologies to combine text and image ML models

### 1. Multiple screenings for text models
There are different methods for categorising our items(data) based on their text description. Instead of using them individually, we decided to use them sequentially.

**1a. String-matching + NN**:
  We deduced that String-Matching is a simple yet highly effective way to categorise items, as long as we set a high threshold. Items who remain unclassified (because of poor text description and thus low score based on the fuzzywuzzy library) will then be predicted using a trained neural network model. 

**1b. Parent Categories**:
  We first predcit the broad categories of the items (eg. Dress, Top) before predicting their specific sub-category (eg. Wedding Dress, T-shirt, Blouse etc.)

### 2. Ensemble Learning
Aggregate the outputs of various text and image classifiers that we have trained.

### 3. Concatenation
Form a new model by concatenating an image and text model, before adding a final layer with an activation function to normalise the results.
