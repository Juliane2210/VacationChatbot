# pip install transformers
# pip install tensorflow_hub
# pip install transformers[torch]
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader, Dataset


import pandas as pd
from sklearn.metrics import precision_score, recall_score, mean_squared_error
from sklearn.metrics import roc_curve, precision_recall_curve, auc, precision_score, recall_score


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import svm
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


from sentence_transformers import SentenceTransformer, util
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel, RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder


import tensorflow_hub as hub
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

######################## SETTING GLOBAL VARIABLES ####################################


# SBERT model
# I use the 'bert-base-nli-mean-tokens' model recommended by ChatGPT: This model uses the BERT-base architecture and has been fine-tuned on the Natural Language Inference (NLI) task.
# It generates sentence embeddings by taking the mean of the output embeddings of all tokens in the sentence.

# m_modelSBERT = SentenceTransformer('all-MiniLM-L6-v2')

m_modelSBERT = SentenceTransformer('bert-base-nli-mean-tokens')


# Initialize the RoBERTa tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model_RoBERTa = RobertaForSequenceClassification.from_pretrained(
    'roberta-base')


m_testDataFile = r"C:\Juliane School\SecondSemester2024\CSI5180\ProgrammingProject\ProjectYesh\Project\Code\SemEval-PIT2015\data\SemEval-PIT2015-github\SemEval-PIT2015-github\data\test.data"
m_devDataFile = r"C:\Juliane School\SecondSemester2024\CSI5180\ProgrammingProject\ProjectYesh\Project\Code\SemEval-PIT2015\data\SemEval-PIT2015-github\SemEval-PIT2015-github\data\dev.data"
m_trainDataFile = r"C:\Juliane School\SecondSemester2024\CSI5180\ProgrammingProject\ProjectYesh\Project\Code\SemEval-PIT2015\data\SemEval-PIT2015-github\SemEval-PIT2015-github\data\train.data"

# Define column names for data
m_columns = ["Topic_Id", "Topic_Name", "Sent_1",
             "Sent_2", "Label", "Sent_1_tag", "Sent_2_tag"]

#################################### HELPER FUNCTIONS #####################################
# Step 1: Load the dataset

#
# Helper to load the dataset
#


def loadDataset(file_path):
    data = pd.read_csv(file_path, sep='\t', header=None, names=m_columns)
    # Strip leading and trailing spaces from all fields
    data = data.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    return data

#
# Ref: https://github.com/cocoxu/SemEval-PIT2015/blob/master/README.md
#


def loadDevDataset(file_path):
    # Read the data file into a DataFrame
    data = loadDataset(file_path)

    # Normalize the data with the following rules
    # paraphrases: (3, 2) (4, 1) (5, 0)
    # non-paraphrases: (1, 4) (0, 5)
    # debatable: (2, 3)  which you may discard if training binary classifier
    data['Label'] = data['Label'].apply(
        lambda x: 1.0 if x.strip() in ["(3, 2)", "(4, 1)", "(5, 0)"] else 0.0)

    return data

#
# The Train and Dev dataset have the same format for the "Label" column
# I re-use the same function within to transform the train data
#


def loadTrainDataset(file_path):
    return loadDevDataset(file_path)


#
# The test dataset has a single numeric entry for the "Label" column
# We need to transform this into a binary Label (0 or 1)
#
def loadTestDataset(file_path):
    data = loadDataset(file_path)
    # Masasage the data with the following rules
    #   The "Label" column for *test data* is in a format of a single digit between
    #   between 0 (no relation) and 5 (semantic equivalence), annotated by expert.
    #   We would suggest map them to binary labels as follows:

    #     paraphrases: 4 or 5
    #     non-paraphrases: 0 or 1 or 2
    #     debatable: 3   which we discarded in Paraphrase Identification evaluation
    data['Label'] = data['Label'].apply(
        lambda x: 1.0 if (x > 3) else 0.0)

    return data


#
# Visualizations
# Helper functions that will be called to perform visualizations of model evaluation scores
#

def visualize_predictions_confusionMatrix(title, true_labels, predicted_labels):
    # Plot confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[
                '0', '1'])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title + ": Confusion Matrix")

    plt.show()


def visualize_predictions_ROC(title, true_labels, predicted_labels):
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(true_labels, predicted_labels)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title + ': Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()


def visualize_predictions_PrecisionRecall(title, true_labels, predicted_labels):
    # Calculate Precision-Recall curve
    precision, recall, _ = precision_recall_curve(
        true_labels, predicted_labels)
    pr_auc = auc(recall, precision)

    # Plot Precision-Recall curve
    plt.figure()
    plt.plot(recall, precision, color='blue', lw=2,
             label='Precision-Recall curve (area = %0.2f)' % pr_auc)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(title + ': Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.show()

 ###################### MODELS #####################


#
# SBERT model helper function
#
def algorithmSBERT(sentence1, sentence2, model):

    embeddings = model.encode([sentence1, sentence2])
    # Ensure embeddings are normalized
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    #  Compute cosine similarity
    similarity_score = cosine_similarity(
        [embeddings[0]], [embeddings[1]])[0][0]

    # print(f"{sentence1} >>>> {sentence2} == {similarity_score}")
    return 1.0 if similarity_score > 0.5 else 0.0


#
# Refined SBERT (Algo A)
# Fine-tune SBERT with training data
#

def fineTuneSBERT(train_data):

    model = m_modelSBERT
    # Convert training data to InputExample objects
    train_examples = [InputExample(
        texts=[row['Sent_1'], row['Sent_2']], label=row['Label']) for _, row in train_data.iterrows()]

    # Create a PyTorch DataLoader for training

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    # Define loss function
    loss_function = losses.CosineSimilarityLoss(model=model)

    # Fine-tune SBERT model on the training data
    model.fit(train_objectives=[(train_dataloader, loss_function)], epochs=1)

    return model

#
# SBERT is pre-trained
# Function to evaluate the Refined SBERT model, which will create visualizations
#


def evaluateSBERTRefined(data, model):
    predictedLabels = [algorithmSBERT(
        row['Sent_1'], row['Sent_2'], model) for _, row in data.iterrows()]

    trueLabels = data['Label']

    precision = precision_score(trueLabels, predictedLabels)
    recall = recall_score(trueLabels, predictedLabels)

    visualize_predictions_confusionMatrix(
        "SBERT Refined (Algo A)", trueLabels, predictedLabels)
    visualize_predictions_ROC(
        "SBERT Refined (Algo A)", trueLabels, predictedLabels)
    visualize_predictions_PrecisionRecall(
        "SBERT Refined (Algo A)", trueLabels, predictedLabels)
    # Print classification report
    print("SBERT Refined (Algo A) Classification Report")
    print(classification_report(trueLabels, predictedLabels, zero_division=1))
    return precision, recall


#
# RoBERTa (Algo B)
#

def encode_sentences_roberta(sentences):
    # Tokenize sentences and prepare them for RoBERTa
    encoded_input = tokenizer(sentences, padding=True,
                              truncation=True, return_tensors='pt')
    # Compute token embeddings
    with torch.no_grad():
        model_output = model_RoBERTa(**encoded_input)
    # Take the mean of the token embeddings to get sentence embeddings
    sentence_embeddings = model_output.last_hidden_state.mean(dim=1)
    return sentence_embeddings


def algorithmRoBERTa(sentence1, sentence2):
    # Encode sentences
    embeddings1 = encode_sentences_roberta([sentence1])[0]
    embeddings2 = encode_sentences_roberta([sentence2])[0]
    # Compute cosine similarity
    similarity_score = torch.cosine_similarity(
        embeddings1, embeddings2, dim=0).item()
    return 1.0 if similarity_score > 0.5 else 0.0


def fine_tune_roberta(model, train_dataset, val_dataset):
    training_args = TrainingArguments(
        output_dir='./results',          # Output directory for model checkpoints
        num_train_epochs=1,              # Number of training epochs
        per_device_train_batch_size=16,  # Batch size for training
        per_device_eval_batch_size=64,   # Batch size for evaluation
        warmup_steps=500,                # Number of warmup steps
        weight_decay=0.01,               # Strength of weight decay
        logging_dir='./logs',            # Directory for storing logs
        logging_steps=10,
        evaluation_strategy="epoch",     # Evaluate after each epoch
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,  # Define your compute_metrics function
    )

    trainer.train()


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

#
# Function to evaluate the RoBERTa model, which will create visualizations
#


def evaluateRoBERTa(devData):
    predictedLabels = [algorithmRoBERTa(
        row['Sent_1'], row['Sent_2']) for _, row in devData.iterrows()]

    trueLabels = devData['Label']

    # Since RoBERTa outputs are already in the form of 0s and 1s, we don't need to transform them further

    precision = precision_score(trueLabels, predictedLabels)
    recall = recall_score(trueLabels, predictedLabels)

    # Visualize predictions with confusion matrix, ROC, and Precision-Recall curves
    visualize_predictions_confusionMatrix(
        "RoBERTa (Algo B)", trueLabels, predictedLabels)
    visualize_predictions_ROC(
        "RoBERTa (Algo B)", trueLabels, predictedLabels)
    visualize_predictions_PrecisionRecall(
        "RoBERTa (Algo B)", trueLabels, predictedLabels)

    # Print classification report
    print("RoBERTa Classification Report")
    print(classification_report(trueLabels, predictedLabels, zero_division=1))

    return precision, recall

 ################################# MAIN #######################################################
if __name__ == "__main__":

    print("*********** Paraphrase Detection ***********")
    print("=====================================")

    devDataset = loadDevDataset(m_devDataFile)
    trainDataset = loadTrainDataset(m_trainDataFile)
    testDataset = loadTestDataset(m_testDataFile)

    # Train the SBERT model to be more accurate
    # model = fineTuneSBERT(trainDataset)
    # evaluateSBERTRefined(devDataset, model)

    # Evaluate RoBERTa
    fine_tune_roberta(model_RoBERTa, trainDataset, devDataset)
    evaluateRoBERTa(devDataset)


############################ CHANGE THIS#############

    #
    # Evaluate the best data set (refined SBERT) on the test dataset.
    #
    # evaluateSBERT(testDataset, model)
