def getAnswer(question):

    #
    # This should load the paraphrase model and return a result based on the question.
    #

    return "The answer is ... Jubs"


def main():
    # Example questions
    questions = [
        "where can I go on vacation",
        "what is spain like this time of year",
        "random question"
    ]

    # Get answers for each question
    for question in questions:
        answer = getAnswer(question)
        print(f"Question: {question}\nAnswer: {answer}\n")


if __name__ == "__main__":
    main()
