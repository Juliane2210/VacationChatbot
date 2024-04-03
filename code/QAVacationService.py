
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import normalize
import os
import pickle

csv_file = './/code//paraphrase_nlp//final_dataset//questions_answers-cooking-train.csv'

embedder_path = './/code//paraphrase_nlp//models//embedder_model'
vectorizer_path = './/code//paraphrase_nlp//models//vectorizer_model.pkl'
bert_model_path = './/code//paraphrase_nlp//models//bert_model.pt'
question_embeddings_path = './/code//paraphrase_nlp//models//question_embeddings.pkl'


def load_embeddings(embeddings_path):
    with open(embeddings_path, 'rb') as f:
        embeddings = pickle.load(f)
    return embeddings


question_embeddings = load_embeddings(question_embeddings_path)


# Function to load embedder, vectorizer, and BERT model and find closest answer
def getAnswer(question):
    # Load embedder
    embedder = SentenceTransformer(embedder_path)

    # Load vectorizer
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)

    # Load BERT model
    model = torch.load(bert_model_path)
    model.eval()

    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Load data
    data = pd.read_csv(csv_file)
    questions = data['Question'].tolist()
    answers = data['Answer'].tolist()

    # Encode input question
    question_embedding = embedder.encode(question)

    # Transform question into vector
    question_vector = vectorizer.transform([question])

    # Compute cosine similarities with question embeddings and question vector
    similarities = cosine_similarity(
        question_embedding.reshape(1, -1), question_embeddings)
    # Compute cosine similarities with question vector
    similarities_vector = cosine_similarity(
        question_vector, vectorizer.transform(questions))

    # Combine similarities from BERT and vectorizer
    combined_similarities = 0.5 * similarities + 0.5 * similarities_vector

    # Find index of most similar question
    closest_index = np.argmax(combined_similarities)

    # Return corresponding answer
    return answers[closest_index]


def main():
    # Example questions
    questions = [
        "what are scallions",
        "what causes my watermelon to rot",
        "random question"
    ]

    # Get answers for each question
    for question in questions:
        answer = getAnswer(question)
        print(f"Question: {question}\nAnswer: {answer}\n")


if __name__ == "__main__":
    main()
