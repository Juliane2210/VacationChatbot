from flask import Flask, render_template, request
from flask_cors import CORS
import QAVacationService
from IntentService import getIntent
import requests

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Create endpoint to host PyChat HTML page


@app.route('/')
def index():
    return render_template('index.html')

# Create endpoint to cater input query of Exploria


def isGreetingIntent(intent):
    if (intent[0] == "greeting"):
        return True
    return False


def isLifestyleIntent(intent):

    #
    # Make a list of all the intents that match to lifestyle
    # Travel, Movie, Cooking
    #
    if (intent[0] == "cook_time"):
        return True
    return False


def wolfram_alpha_query(query):
    # Wolfram Alpha API endpoint
    # url = "http://api.wolframalpha.com/v1/result"
    url = "http://api.wolframalpha.com/v1/spoken"

    # Parameters for the query
    params = {
        "appid": "QG759U-K96T398GRW",
        "i": query
    }

    # Make the API call
    response = requests.get(url, params=params)

    # Check if the request was successful
    if response.status_code == 200:
        return response.text
    else:
        print("Error making API call: ", response.text)
        return "Please reformulate your question."


@app.route('/submit', methods=["GET", "POST"])
def processInputQuery():
    req = request.get_json()
    if req is not None and "msg" in req:

        utterance = req["msg"]

        #
        # The intent service could be remotely hosted
        #
        intent = getIntent(utterance)

        intentText = intent[0]

        response = " out of scope buddy bot "
        if (isLifestyleIntent(intent)):

            response = QAVacationService.getAnswer(req["msg"])

            response = "I think you are asking about " + intentText + ". " + response
        elif (isGreetingIntent(intent)):
            #
            # Go to a weather API
            #
            response = " Hello this is Exploria.  Ask me any question."
        else:

            #
            # General knowledge query will go to wikidata
            #
            response = wolfram_alpha_query(utterance)

            response = "I think you are asking about " + intentText + ". " + response

        return response
    else:
        return "Invalid input"


# Main to run the server on specific port
if __name__ == '__main__':
    app.run(debug=False, port=2412)
