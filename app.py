from flask import Flask, render_template, request, jsonify
import nltk
import numpy as np
import os
import random
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Ensure required NLTK resources are available
import os
#nltk.data.path.append(os.path.join(os.getenv("APPDATA"), "nltk_data"))
nltk.data.path.append(os.path.expanduser("~/nltk_data"))


nltk.download('punkt_tab')
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt_tab')
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# File path
file_path = 'chatbot.txt'

# Initialize sentence tokens
sentence_tokens = []
word_tokens = []

# Read chatbot.txt file
try:
    if not os.path.exists(file_path):
        raise FileNotFoundError("Error: chatbot.txt not found. Please check the file path.")

    with open(file_path, 'r', errors='ignore') as chatbots_file:
        content = chatbots_file.read().strip().lower()

    if not content:
        raise ValueError("Error: chatbot.txt is empty or unreadable.")

    # Tokenize sentences and words
    sentence_tokens = nltk.sent_tokenize(content)
    word_tokens = nltk.word_tokenize(content)

except FileNotFoundError as e:
    print(e)
except ValueError as e:
    print(e)
except Exception as e:
    print("❌ An unexpected error occurred:", str(e))

# Lemmatization
lemmer = nltk.stem.WordNetLemmatizer()

def lem_tokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def lem_normalize(text):
    return lem_tokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

# Greeting Responses
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey")
GREETING_RESPONSES = ["hi", "hey", "hello", "hi there!", "I'm glad to chat with you!"]

def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)
    return None

# Chatbot Response Function
def get_response(user_response):
    global sentence_tokens  # Ensure we use the global list

    if not sentence_tokens:  # Handle case where chatbot.txt wasn't loaded
        return "I'm sorry! My knowledge base is empty."

    sentence_tokens.append(user_response)

    vectorizer = TfidfVectorizer(tokenizer=lem_normalize, stop_words='english')

    try:
        tfidf = vectorizer.fit_transform(sentence_tokens)
        similarity = cosine_similarity(tfidf[-1], tfidf)

        idx = similarity.argsort()[0][-2]
        flat = similarity.flatten()
        flat.sort()

        response_score = flat[-2]
        if response_score == 0:
            robo_response = "I'm sorry! I don't understand."
        else:
            robo_response = sentence_tokens[idx]

    except Exception as e:
        robo_response = "I encountered an error processing your request."

    sentence_tokens.pop()  # Remove last user input
    return robo_response

# Flask Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message', "").strip()
    
    if not user_input:
        return jsonify({'response': "Please enter a valid message."})

    bot_response = greeting(user_input) or get_response(user_input)
    return jsonify({'response': bot_response})

if __name__ == '__main__':
    app.run(debug=True)
