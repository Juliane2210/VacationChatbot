import pandas as pd
import pickle
import warnings

warnings.filterwarnings('ignore')

# Load model
sgd = pickle.load(open('.//code//intent//models//SGD_Inscope.pkl', 'rb'))
cv_in = pickle.load(
    open('.//code//intent//models//CountVect_Inscope.pkl', 'rb'))
encoder = pickle.load(open('.//code//intent//models//LabelEncoder.pkl', 'rb'))


def getIntent(utterance):
    # Convert data into dataframe
    data_df = pd.DataFrame({'text': [utterance]})
    X = cv_in.transform(data_df.text)

    # Predict
    result = sgd.predict(X)
    result = encoder.inverse_transform(result)

    return result


def main():
    # Example calls to getIntent function
    intents = [
        "what is the cheapest vacation",
        "where to buy red t-shirt",
        "10-4",
        "how long should jello stay in the fridge",
        "how long will jello last",
        "what time is it",
        "how to cook jello",
        "what is the weather"
    ]

    for intent in intents:
        print(getIntent(intent))


if __name__ == "__main__":
    main()
