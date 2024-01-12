# Reddit Age Prediction Project

## Overview

This project is part of the course "Language and AI" within the Data Science bachelor of the Technical University of Eindhoven and Tilburg University.
This project is designed to predict the age of Reddit users based on their posts and comments using natural language processing (NLP) techniques. The project consists of several Python modules, each with a distinct role in the data processing and analysis pipeline.

## Project Structure

-   `Analysis.py`: This is the main entry point of the project. It orchestrates the entire data analysis process by utilizing other modules in the project. To run the full analysis pipeline, execute this script.

-   `DataPreparation.py`: Contains routines for initial data setup, such as loading data from various sources and initial data cleaning steps that prepare the data for further processing.

-   `DataPreprocessor.py`: Defines the `DataPreprocessor` class responsible for more advanced data cleaning processes, such as tokenization and removal of frequent words, especially in preparation for vectorization.

-   `DataVisualizer.py`: Implements the `DataVisualizer` class, which provides visualization functions to explore the data and to understand the distribution of features within the data.

-   `ModelCollection.py`: Contains a collection of machine learning models and the corresponding functions to train, test, and evaluate these models on the processed data.

-   `TFIDFTransformer.py`: Defines the `TFIDFTransformer` class that transforms text data into TF-IDF vectors, which are utilized by the machine learning models for prediction.

-   `EvaluationMetrics.csv`: A CSV file that stores the performance metrics of the various models used in the project. This file is updated after model evaluation.

-   `Results`: A directory that stores the output of the analysis, such as model predictions and figures from data visualizations.

## Running the Analysis

To run the analysis pipeline, ensure that all dependencies are installed, and then execute the `Analysis.py` script, which will use the other modules as needed. The script will guide you through data preprocessing, feature extraction with TF-IDF, model training, and evaluation, and will save the results in the `Results` directory.

## Dependencies

To run this project, it is necessary to have installed the following dependencies:

-   `matplotlib`
-   `scipy`
-   `pandas`
-   `sklearn` (scikit-learn)
-   `seaborn`
-   `numpy`
-   `Ipython`
-   `nltk` (Natural Language Toolkit)
-   `re` (Regular Expressions)
