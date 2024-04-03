import requests

# Function that will call API to retirve answer to query


def getAnswer(query):
    # Wolfram Alpha API endpoint endpoint for spoken responses ('/spoken')
    # url = "http://api.wolframalpha.com/v1/result"
    url = "http://api.wolframalpha.com/v1/spoken"

    # Parameters for the API request: application's API key and the input query
    params = {
        "appid": "QG759U-K96T398GRW",
        "i": query
    }

    # Make the API call : GET request with the API url and request parameters
    response = requests.get(url, params=params)

    # Check if the request was successful: 200 (HTTP OK)
    if response.status_code == 200:
        # Return the text of the response, which contains the answer to the query
        return response.text
    else:
        # If the request was not successful, print an error message including the response text
        print("Error making API call: ", response.text)
        # Return a message prompting the user to reformulate their question
        return "Please reformulate your question."


# main function to test code
def main():
    # Example questions
    questions = [
        "what is spaghetti",
        "what is the weather in Ottawa",
        "random question"
    ]

    # Get answers for each question
    for question in questions:
        answer = getAnswer(question)
        print(f"Question: {question}\nAnswer: {answer}\n")


if __name__ == "__main__":
    main()
