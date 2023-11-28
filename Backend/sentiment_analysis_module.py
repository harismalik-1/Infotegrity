import re
import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from politicalBiases import left_words, center_words, right_words
from scipy.special import softmax
import spacy
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import textstat
import logging

# Ensure NLTK dependencies are downloaded
nltk.download('punkt')
nltk.download('stopwords')

nlp = spacy.load('en_core_web_sm')

def fetch_and_clean_text(url):
    # Fetching text from URL
    r = requests.get(url)
    r.encoding = 'utf-8'
    html = r.text
    soup = BeautifulSoup(html, 'html.parser')
    text = soup.get_text()
    title = soup.find('title').text if soup.find('title') else 'No Title Found'

    # Cleaning text
    clean_text = text.replace("\n", " ").replace("/", " ")
    clean_text = ''.join([c for c in clean_text if c not in ["'", "\""]])
    return title, clean_text

def highlight_political_bias(text, word_lists):
    doc = nlp(text)
    highlighted_sentences = []
    index = 1

    # Mapping words to categories
    word_to_category = {word: 'left' for word in left_words}
    word_to_category.update({word: 'center' for word in center_words})
    word_to_category.update({word: 'right' for word in right_words})

    all_words = sum(word_lists, [])
    pattern = re.compile(r'\b(' + '|'.join(re.escape(word) for word in all_words) + r')\b', re.IGNORECASE)

    for sent in doc.sents:
        sentence = sent.text
        matches = pattern.finditer(sentence)
        for match in matches:
            start, end = match.span()
            word = sentence[start:end].lower()
            category = word_to_category.get(word, 'unknown')
            highlighted_sentence = f"{index}. Category: {category.title()}\n Sentence: '{sentence.strip()}'\n"
            highlighted_sentences.append(highlighted_sentence)
            index += 1
    
    return highlighted_sentences


def preprocess(text):
    # Preprocess text for RoBERTa model
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

def analyze_sentiment_roberta(text):
    # RoBERTa sentiment analysis
    MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    config = AutoConfig.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)

    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    scores = softmax(output[0][0].detach().numpy())

    ranking = np.argsort(scores)[::-1]
    results = {}
    for i in range(scores.shape[0]):
        label = config.id2label[ranking[i]]
        score = np.round(float(scores[ranking[i]]), 4)
        results[label] = score
    return results

def analyze_sentiment_textblob(text):
    # TextBlob sentiment analysis
    tokens = nlp(text)
    sentences = [sent.text.strip() for sent in tokens.sents]

    sentiments = []
    for s in sentences:
        txt = TextBlob(s)
        sentiments.append({
            'sentence': s,
            'polarity': txt.sentiment.polarity,
            'subjectivity': txt.sentiment.subjectivity
        })
    return sentiments

def get_dominant_political_bias(word_counts):
    dominant_bias = max(word_counts, key=word_counts.get)
    return dominant_bias

def analyze_word_frequency(text):
    # Analyze word frequency
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+|[.,]')
    tokens = tokenizer.tokenize(text.lower())
    filtered_words = [word for word in tokens if word not in stopwords.words('english')]
    
    freq_dist = nltk.FreqDist(filtered_words)
    return freq_dist.most_common(20)

def count_political_words(text, word_lists):
    word_counts = { "left": 0, "center": 0, "right": 0 }
    for word in text.split():
        word = word.lower()
        if word in left_words:
            word_counts["far_left"] += 1
        elif word in center_words:
            word_counts["center"] += 1
        elif word in right_words:
            word_counts["right"] += 1
    return word_counts

def named_entity_recognition(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

def text_complexity_analysis(text):
    flesch_reading_ease = textstat.flesch_reading_ease(text)
    smog_index = textstat.smog_index(text)
    return {'flesch_reading_ease': flesch_reading_ease, 'smog_index': smog_index}

logging.getLogger('transformers').setLevel(logging.ERROR)

def analyze_url(url):
    title, clean_text = fetch_and_clean_text(url)
    roberta_results = analyze_sentiment_roberta(preprocess(clean_text[:500]))
    textblob_results = analyze_sentiment_textblob(clean_text)
    named_entities = named_entity_recognition(clean_text)
    complexity = text_complexity_analysis(clean_text)
    word_lists = [left_words, center_words, right_words]
    political_bias_sentences = highlight_political_bias(clean_text, word_lists)
    political_word_counts = count_political_words(clean_text, word_lists)
    dominant_bias = get_dominant_political_bias(political_word_counts)

    # Format the output
    output = f"\nArticle Title: {title}\n\n"
    output += "Sentiment Analysis Results:\n\n"
    output += "RoBERTa Model Sentiment Scores:\n"
    for sentiment, score in roberta_results.items():
        output += f" - {sentiment.title()}: {round(score, 4)}\n"

    output += "\nTextBlob Model Analysis (Top 5 Sentences):\n"
    for result in textblob_results[:5]:
        sentence = (result['sentence'][:75] + '...') if len(result['sentence']) > 75 else result['sentence']
        output += f" - Sentence: {sentence}\n"
        output += f"   Polarity: {round(result['polarity'], 2)}, Subjectivity: {round(result['subjectivity'], 2)}\n"

    output += "\nPolitical Bias Analysis (Highlighted Sentences):\n"
    for highlighted_sentence in political_bias_sentences:
        output += f"{highlighted_sentence}\n"

    output += "\nOverall Political Bias Assessment:\n"
    output += f" - Dominant Political Bias: {dominant_bias.title()}\n"

    # output += " - Word Count Breakdown: " + ", ".join([f"{key.title()}: {count}" for key, count in political_word_counts.items()]) + "\n"

    output += "\nNamed Entities (Top 5):\n"
    for entity, label in named_entities[:5]:
        output += f" - {entity} ({label})\n"
        
    # output += "\nText Complexity Analysis:\n"
    # output += f" - Flesch Reading Ease: {complexity['flesch_reading_ease']}/100\n"
    # output += f" - SMOG Index: {complexity['smog_index']} - a person in grade {complexity['smog_index']} can understand in first read\n"

    return output

