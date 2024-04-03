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

# Function to train and save the embedder, vectorizer, and BERT model


def train_and_save_model(csv_file, embedder_path, vectorizer_path, bert_model_path, question_embeddings_path):
    # Load data
    data = pd.read_csv(csv_file)

    # Initialize BERT model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval()

    # Initialize Sentence Transformer
    embedder = SentenceTransformer('paraphrase-distilroberta-base-v2')

    # Extract questions and answers
    questions = data['Question'].tolist()
    answers = data['Answer'].tolist()

    # Train vectorizer
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(questions)

    # Save embedder
    embedder.save(embedder_path)

    # Save vectorizer
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)

    # Save BERT model
    torch.save(model, bert_model_path)

    question_embeddings = compute_embeddings(bert_model_path, csv_file)
    with open(question_embeddings_path, 'wb') as f:
        pickle.dump(question_embeddings, f)


# Function to load embedder, vectorizer, and BERT model and find closest answer


def compute_embeddings(bert_model_path, csv_file):

    # Load BERT model
    model = torch.load(bert_model_path)
    model.eval()

    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Load data
    data = pd.read_csv(csv_file)
    questions = data['Question'].tolist()

    # Compute BERT embeddings for questions
    question_embeddings = []
    for q in questions:
        inputs = tokenizer(q, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state.mean(
                dim=1).squeeze().numpy()
            question_embeddings.append(embeddings)

    return question_embeddings


# Example usage
if __name__ == "__main__":

    csv_file = './/code//paraphrase_nlp//final_dataset//questions_answers-cooking-train.csv'

    embedder_path = './/code//paraphrase_nlp//models//embedder_model'
    vectorizer_path = './/code//paraphrase_nlp//models//vectorizer_model.pkl'
    bert_model_path = './/code//paraphrase_nlp//models//bert_model.pt'
    question_embeddings_path = './/code//paraphrase_nlp//models//question_embeddings.pkl'

    # Train and save the model
    train_and_save_model(csv_file, embedder_path,
                         vectorizer_path, bert_model_path, question_embeddings_path)

    # question_embeddings = compute_embeddings(bert_model_path, csv_file)

    # question_embeddings = load_embeddings(question_embeddings_path)

    # print('\n\n***************Exploria***************\n')
    # print('Welcome to the Exploria Console.')
    # print('\n***************Exploria***************\n\n')
    # while True:
    #     print('===============================================================================\n')
    #     question = input('Input Query:\n')

    #     # response, confident = GenerateResponse(query)

    #     answer = getAnswer(question_embeddings, question, embedder_path,
    #                        vectorizer_path, bert_model_path, csv_file)

    #     # print(f"Confident: {confident}%")
    #     print("\nThe reponse:\n--------------\n")
    #     print(answer)
