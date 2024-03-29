from flask import Flask, render_template, request
from flask_cors import CORS
import QAVacationService
import WolframAlphaService
from IntentService import getIntent
import json

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes


@app.route('/')
def index():
    return render_template('index.html')

# Create endpoint to cater input query of Exploria


def isGreetingIntent(intent_classification):
    if (intent_classification == "greeting"):
        return True
    return False


def isLifestyleIntent(intent_classification):
    #
    # Make a list of all the intents that match to lifestyle
    # Travel, Movie, Cooking
    #
    if (intent_classification == "cook_time"):
        return True
    return False


@app.route('/submit', methods=["GET", "POST"])
def processInputQuery():
    req = request.get_json()
    if req is not None and "msg" in req:

        utterance = req["msg"]

        #
        # The intent service could be remotely hosted and returns the intent classification with a confidence score.
        #
        intent_json = json.loads(getIntent(utterance))
        intent_classification = intent_json["intent"]
        confidence_score = intent_json["confidence"]

        if (confidence_score < 20):
            return "I didn't understand, please rephrase your question."

        response = f"I am {confidence_score} confident you asked about {intent_classification} "
        if (isLifestyleIntent(intent_classification)):
            #
            # Specialized responses for custom category
            #
            answer = QAVacationService.getAnswer(utterance)
            response = response + ".  " + answer

        elif (isGreetingIntent(intent_classification)):
            #
            # Custom Greeting
            #
            response = "Hello this is Exploria.  Ask me any question."
        else:
            #
            # General knowledge query will go to WolframAlpha
            #
            answer = WolframAlphaService.getAnswer(utterance)
            response = response + ".  " + answer

        return response
    else:
        return "Invalid input"


# Main to run the server on specific port
if __name__ == '__main__':
    app.run(debug=False, port=2412)
