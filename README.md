# README: Sentiment Analysis of Amazon Cell Phone Reviews Using Machine Learning Models

## Project Overview
This project focuses on **sentiment analysis** of Amazon mobile phone reviews using machine learning models. The objective is to classify reviews as either **positive** or **negative** to provide insights into customer satisfaction and support e-commerce decision-making.

## Key Features
1. **Dataset**:
   - Source: Kaggle ([Amazon Cell Phones Reviews](https://www.kaggle.com/datasets/nibras/amazon-cell-phone-reviews)).
   - Contains 67,986 reviews with features like text, rating, and product brand.
   - Balanced dataset created with 5,000 positive and 5,000 negative reviews.

2. **Preprocessing**:
   - Tokenization, stopword removal, stemming, and punctuation cleaning.
   - Text transformed into numerical features using **TF-IDF vectorization**.
   - Sentiments labeled as positive (rating > 3) or negative (rating ≤ 3).

3. **Models Implemented**:
   - **Support Vector Machine (SVM):** Achieved the highest accuracy (89.15%).
   - **Random Forest Classifier:** Robust but slightly lower accuracy (86.4%).
   - **Recurrent Neural Network (RNN):** Sequential data processing but suffered from overfitting (78% accuracy).
   - **Long Short-Term Memory (LSTM):** Outperformed RNN with better handling of sequential dependencies (87% accuracy).

4. **Performance Metrics**:
   - Accuracy, Precision, Recall, and F1 Score.
   - Confusion matrices visualized for SVM and Random Forest.

## Tools and Libraries
- **Python Libraries**: `pandas`, `nltk`, `sklearn`, `tensorflow`, `matplotlib`, `seaborn`, `wordcloud`.
- **Machine Learning Frameworks**: Scikit-learn, TensorFlow/Keras.



## How to Run the Project
1. **Setup Environment**:
   - Use Google Colab to run the project.
   - Upload the dataset file amazon-cell-phone-reviews.csv to your Colab session.

2. **Preprocess and Train Models**:
   - Run the provided single notebook file to clean, balance, transform the dataset, and train models (SVM, Random Forest, RNN, and LSTM).


## Key Results
| Model               | Accuracy | Precision | Recall | F1 Score |
|---------------------|----------|-----------|--------|----------|
| Support Vector Machine (SVM) | 89.15%   | 89.18%    | 89.15% | 89.15%   |
| Random Forest       | 86.40%   | 86.52%    | 86.40% | 86.39%   |
| Recurrent Neural Network (RNN) | 78%      | -         | -      | -        |
| Long Short-Term Memory (LSTM) | 87%      | -         | -      | -        |

## Insights and Applications
1. **Business Applications**:
   - Understand customer satisfaction and concerns.
   - Inform marketing strategies by highlighting commonly appreciated product features.
   - Identify and resolve frequent customer complaints for product improvement.

2. **Research Extensions**:
   - Explore multiclass sentiment analysis (e.g., positive, negative, neutral).
   - Experiment with advanced feature extraction techniques (e.g., Word2Vec, GloVe).
   - Combine traditional models (e.g., SVM) with deep learning approaches (e.g., LSTM).

## Limitations
- Imbalanced original dataset required balancing.
- Binary classification restricts nuanced sentiment analysis.
- RNN and LSTM models were computationally intensive and prone to overfitting.

## Future Work
- Implementing hybrid models to combine the strengths of traditional and deep learning techniques.
- Expanding to multiclass sentiment analysis.
- Incorporating real-time analysis for scalability in live e-commerce platforms.
