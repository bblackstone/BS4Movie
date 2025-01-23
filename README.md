# Movie Recommendation System

This project implements a movie recommendation system using collaborative filtering and content-based filtering techniques. It leverages the power of machine learning and natural language processing to provide personalized movie recommendations to users.

## Features

* **Content-based Filtering:** Recommends movies similar to those the user has liked in the past based on their genre, theme, and style.
* **Collaborative Filtering:** Recommends movies based on the preferences of similar users.
* **TF-IDF Vectorization:** Converts movie overviews into numerical representations for similarity calculations.
* **Cosine Similarity:** Measures the similarity between movies based on their TF-IDF vectors.
* **Firefly Algorithm:** Optimizes the TF-IDF weights for better recommendations.
* **Nearest Neighbors:** Finds the most similar movies to a given movie.
* **Phi-2/Llama-3.1-8B-Instruct:** Generates natural language explanations for the recommendations.
* **Streamlit:** Creates an interactive web application for users to input their preferences and receive recommendations.
* **Fuzzy Matching:** Handles minor spelling variations in movie titles.
* **Genre Distribution:** Provides insights into the popularity of different movie genres.
* **Popularity vs. Vote Average:** Explores the relationship between movie popularity and ratings.
* **Release Year Distribution:** Visualizes the distribution of movie releases over time.

## Requirements

* Python 3.8 or higher
* Google Colab or Jupyter Notebook
* Libraries: scikit-learn, transformers, torch, sentencepiece, fuzzywuzzy, python-Levenshtein, streamlit, pyngrok, cuml-cu11

## Installation

1. Clone this repository.
2. Install the necessary libraries using `pip install -r requirements.txt`.
3. Obtain a Hugging Face API token and set it as an environment variable.
4. Obtain a ngrok API token and set it as an environment variable.

## Usage

1. Open the `Movie_Recommendation_System.ipynb` notebook in Google Colab or Jupyter Notebook.
2. Run the cells in the notebook to load the dataset, preprocess the data, train the model, and build the web application.
3. Use the Streamlit interface to input your movie preferences and receive recommendations.

## Acknowledgements

* The movie dataset used in this project is from Kaggle.
* The TF-IDF vectorization, cosine similarity, and nearest neighbors algorithms are implemented using scikit-learn.
* The Firefly Algorithm is implemented using custom Python code.
* The Phi-2/Llama-3.1-8B-Instruct language model is from Hugging Face.
* The Streamlit web framework is used to build the interactive application.

## Contributing

Contributions are welcome! If you find any bugs or have suggestions for improvement, please open an issue or submit a pull request.
