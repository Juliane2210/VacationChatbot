import json
import csv


FILE_COOKING = './/code//paraphrase_nlp//final_dataset//doqa-cooking-train-v2.1.json'
FILE_MOVIES = './/code//paraphrase_nlp//final_dataset//doqa-movies-test-v2.1.json'
FILE_TRAVEL = './/code//paraphrase_nlp//final_dataset//doqa-travel-test-v2.1.json'

OUTPUT_QA = './/code//paraphrase_nlp//final_dataset//questions_answers-cooking-train.csv'


def getQAData(file_path):
    # Load JSON data
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Extract questions and answers
    qa_data = []
    for item in data['data']:
        title = item['title']
        for paragraph in item['paragraphs']:
            for qa in paragraph['qas']:
                question = qa['question']
                # Assuming there is only one answer
                answer = qa['answers'][0]['text']
                qa_data.append(
                    {'Title': title, 'Question': question, 'Answer': answer})
    return qa_data


def saveQAData(qa_data, file_path):
    # Write to CSV
    with open(file_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['Title', 'Question', 'Answer']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for qa in qa_data:
            writer.writerow(qa)


# Example usage
if __name__ == "__main__":

    all_qa_data = []

    qa_data_cooking = getQAData(FILE_COOKING)
    all_qa_data.extend(qa_data_cooking)

    qa_data_movies = getQAData(FILE_MOVIES)
    all_qa_data.extend(qa_data_movies)

    qa_data_travel = getQAData(FILE_TRAVEL)
    all_qa_data.extend(qa_data_travel)

    saveQAData(all_qa_data, OUTPUT_QA)
