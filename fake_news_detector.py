import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle

# Load true and fake news datasets
true_df = pd.read_csv("E:/fake_news_detection_app/news/True.csv")
fake_df = pd.read_csv("E:/fake_news_detection_app/news/Fake.csv")

# Add labels
true_df['label'] = 'REAL'
fake_df['label'] = 'FAKE'

# Combine and shuffle
df = pd.concat([true_df, fake_df], ignore_index=True)
df = df.dropna().sample(frac=1, random_state=42)  # Shuffle the data

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

print(df['label'].value_counts())


# # Vectorize text
# vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
# X_train_tfidf = vectorizer.fit_transform(X_train)
# X_test_tfidf = vectorizer.transform(X_test)

# # Train model
# model = PassiveAggressiveClassifier()
# model.fit(X_train_tfidf, y_train)

# # Evaluate
# y_pred = model.predict(X_test_tfidf)
# print("Accuracy:", accuracy_score(y_test, y_pred))
# print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

#pickling
#pickle.dump(model, open("model.pkl", "wb"))
#pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
