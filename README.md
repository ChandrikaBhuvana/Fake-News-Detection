# 📰 Fake News Detection App

This is a **simple end-to-end machine learning project** that detects whether a news article is **Real** or **Fake**. It is designed to help understand the **complete ML workflow** from data preprocessing and model training to building an interactive web app and deploying it.

## 🎯 Project Overview

The goal of this project is to create a minimal yet functional **fake news classifier** and deploy it using **Streamlit**, making it accessible to anyone without needing to run Python code manually.

It demonstrates key ML and deployment skills, including:

- Data preprocessing with pandas
- Text vectorization using **TF-IDF**
- Training a fast and interpretable model: **PassiveAggressiveClassifier**
- Model evaluation and saving (`pickle`)
- **Building a web app** using Streamlit
- Running and deploying the app locally or on platforms like **Streamlit Cloud**

## 🧠 Machine Learning Skills Showcased

- **Text classification** using natural language processing (NLP)
- Feature extraction using `TfidfVectorizer`
- Model training with scikit-learn
- Accuracy measurement and confusion matrix
- Serialization of model and vectorizer (`pickle`)
- Loading models for real-time inference

## 🚀 Deployment

This project uses **Streamlit** to create an intuitive web interface where users can paste a news article and get instant predictions.

- You can run it locally with:
  ```bash
  streamlit run app.py
  ```
- Or deploy it to [Streamlit Cloud](https://streamlit.io/cloud) for free, with just a GitHub repo and one click.

## 🛠️ How to Use

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/fake-news-detector.git
   cd fake-news-detector
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Launch the app:
   ```bash
   streamlit run app.py
   ```

4. Paste any news article text to see if it’s **REAL** or **FAKE**.

## 📂 Project Structure

```
fake_news_detection_app/
├── app.py                  # Streamlit frontend
├── train_model.py          # Model training and saving
├── model.pkl               # Trained model
├── vectorizer.pkl          # TF-IDF vectorizer
├── news/
│   ├── True.csv            # Real news data
│   └── Fake.csv            # Fake news data
└── README.md               # This file
```

## 🧪 Dataset

The model was trained on the Fake and Real News Dataset from Kaggle:

🔗 [Fake and Real News Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)

## ✅ Why This Project?

This project is ideal for:
- Beginners learning how ML projects are built and deployed
- Demonstrating core skills in resumes or portfolios
- Gaining practical experience with NLP and model deployment

## ❤️ Credits

- Built with [Streamlit](https://streamlit.io/)
- Trained using [scikit-learn](https://scikit-learn.org/)
- Dataset from [Kaggle](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)

---

> **Made with ❤️ to learn and share the basics of ML + deployment**
