https://commitsapporiskushpatel-ek2a94mzpef67ljhhzdsvc.streamlit.app/
# Email Spam Messages Classifier
This project implements a spam message detection system using machine learning. The goal is to classify text messages as either "non-spam" (legitimate) or "spam" (unwanted).

## Dataset

The dataset used in this project is the "SMS Spam Collection Dataset" which can be found [here](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection). It contains a collection of SMS messages labeled as either ham or spam.

## Project Steps

The project follows these steps:

1.  **Data Loading and Preprocessing:**
    *   Load the dataset.
    *   Handle missing values.
    *   Remove duplicate entries.
    *   Rename columns for better readability.
    *   Convert the target variable ('spam'/'ham') into numerical representation (1/0).
    *   Clean the text messages by:
        *   Converting to lowercase.
        *   Removing punctuation.
        *   Tokenizing the messages.
        *   Removing stopwords.
        *   Applying stemming.

2.  **Exploratory Data Analysis (EDA):**
    *   Visualize the most frequent words in spam and ham messages using word clouds.

3.  **Feature Engineering:**
    *   Convert the cleaned text messages into numerical features using the TF-IDF vectorizer.

4.  **Model Training and Evaluation:**
    *   Split the data into training and testing sets.
    *   Train and evaluate three different Naive Bayes models:
        *   Multinomial Naive Bayes
        *   Gaussian Naive Bayes
        *   Bernoulli Naive Bayes
    *   Evaluate the models using metrics such as accuracy, precision, recall, F1-score, and confusion matrices.

5.  **Model Saving:**
    *   Save the best performing model and the TF-IDF vectorizer for future use.

## Dependencies

The following libraries are required to run this project:

*   numpy
*   pandas
*   matplotlib
*   seaborn
*   nltk
*   sklearn
*   wordcloud
*   joblib
## Live Demo
https://commitsapporiskushpatel-ek2a94mzpef67ljhhzdsvc.streamlit.app/
