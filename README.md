# Fake News Detection

## Overview
This project is a **Fake News Detection** system using **Machine Learning**. It classifies news articles as either **real** or **fake** based on text analysis. The model is trained using the **ISOT Fake News Dataset** and uses **TF-IDF Vectorization** with a **Logistic Regression classifier**.

## Features
- Preprocesses news articles (removes punctuation, converts to lowercase, etc.)
- Converts text into numerical features using **TF-IDF**
- Trains a **Logistic Regression** model
- Supports **new text predictions** to determine if news is real or fake
- Provides a **web interface** using **Streamlit**

## Dataset
The project uses the **ISOT Fake News Dataset**, which contains:
- `Fake.csv`: Fake news articles
- `True.csv`: Real news articles

Both files have the following columns:
- `title`: The headline of the news article
- `text`: The full article text
- `subject`: The category of the news
- `date`: The publication date

A `label` column is added to mark **1 for Fake News** and **0 for Real News**.

## Installation
### 1. Clone the Repository
```bash
git clone https://github.com/AbhinandanJain5/FakeNewsDetector.git
cd FakeNewsDetector
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the Model
Run the training script to generate `model.pkl` and `vectorizer.pkl`:
```bash
python train_model.py
```

### 4. Run the Web App
Start the Streamlit app:
```bash
streamlit run app.py
```

## Usage
1. Enter a news article in the web interface.
2. Click the "Predict" button.
3. The system will classify the news as **Real or Fake**.

## File Structure
```
├── Fake.csv              # Fake news dataset
├── True.csv              # Real news dataset
├── train_model.py        # Training script
├── app.py                # Streamlit web app
├── model.pkl             # Trained model
├── vectorizer.pkl        # TF-IDF vectorizer
├── requirements.txt      # Required Python packages
├── README.md             # Project documentation
```

## Future Improvements
- Experiment with **Random Forest** or **SVM** for better accuracy.
- Improve text preprocessing to retain key information.
- Expand dataset to improve model generalization.
- Deploy the web app using **Heroku** or **Streamlit Cloud**.

## License
This project is open-source under the **MIT License**.

---
### Feel free to contribute and improve the project!


