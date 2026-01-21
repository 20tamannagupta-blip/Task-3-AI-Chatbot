
import json
import random
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


nltk.download('punkt')
nltk.download('punkt_tab')

# Load intents file
with open("intents.json", "r", encoding="utf-8") as file:
    data = json.load(file)

patterns = []
tags = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        patterns.append(pattern)
        tags.append(intent["tag"])

# Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(patterns)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X, tags)

# Chat function
def chatbot_response(text):
    text_vector = vectorizer.transform([text])
    tag = model.predict(text_vector)[0]

    for intent in data["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])

    return "Sorry, I didn't understand that."

print("🤖 Chatbot is running! (type 'exit' to stop)")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Bot: Goodbye!")
        break

    response = chatbot_response(user_input)
    print("Bot:", response)
