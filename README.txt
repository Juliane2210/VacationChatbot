__PROJECT DESCRIPTION__

This project is a question answer chatbot called Exploria that can answer questions about cooking, movies, travel, and general knowledge.
It serves as the final submission for the CSI-5180: Topics in AI (Virtual Assistant) course.
The chatbot works by taking in an input question (keyboard or mic) as an utterance from which it will detect an intent 
(with a certain degree of confidence) and depending on the intent, will pass on the utterance to an API (WolframAlpha) or a paraphrase detection service 
that will return an appropriate response.

To provide a good user interface for the chatbot I created a HTML, CSS and JavaScript based web application.


The project is separated in 4 folders under the 'code' main folder: 'intent', 'paraphrase_nlp', 'static' and 'templates'.
Furthermore, the 'code' main folder contains 4 python files: 'ExploriaFlaskAgent.py', 'IntentService.py', 'QAVacationService.py' and 'WolframAlphaService.py'.
The 4 files correspond to the agent and 3 services used in the project.
The 'intent' folder is used for intent detection model creation and contains 2 folders and a python file: 'intent_data' {which contains the dataset used to train the model}, 'models' {which contains the saved model} and 'CreateIntentModels.py' {with the necessary code to train and output the paraphrase detection model}. 
The 'paraphrase_nlp' folder is used for paraphrase detection model creation and contains 2 folders and 2 python files : 'final_dataset' {which contains the dataset used to train the model}, 'models' {which contains the saved model} , 'CleanData.py' {to preprocess the training data} and 'CreateParaphraseModel.py' {with the necessary code to train and output the paraphrase detection model}.



__DEPENDENCIES__


1. Python 3.11.5 
2. Python libraries:
    a. flask (Flask, render_template, request)
    b. flask_cors (CORS)
    c. IntentService (getIntent)
    d. QAVacationService
    e. WolframAlphaService
    f. pandas (pd)
    g. pickle
    h. warnings
    i. numpy (np)
    j. json
    k. requests
    l. torch
    m. sentence_transformers (SentenceTransformer, InputExample, losses)
    n. torch.utils.data (DataLoader, Dataset)
    o. sklearn.metrics (precision_score, recall_score, mean_squared_error, roc_curve, precision_recall_curve, auc, confusion_matrix, classification_report)
    p. sklearn.feature_extraction.text (CountVectorizer)
    q. sklearn.naive_bayes (MultinomialNB)
    r. sklearn.model_selection (train_test_split)
    s. sklearn.linear_model (LogisticRegression)
    t. sklearn.metrics.pairwise (cosine_similarity)
    u. sklearn (svm)
    v. sklearn.datasets (make_classification)
    w. seaborn (sns)
    x. transformers (BertTokenizer, BertModel, AutoTokenizer, AutoModel, RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments)
    y. sklearn.preprocessing (LabelEncoder)
    z. tensorflow_hub (hub)
    aa. scipy.sparse (sp)
    ab. matplotlib.pyplot (plt)
3. HTML/CSS/JavaScript

__STEPS TO RUN CHATBOT__

{Steps 4-6 aren't necessary if you want to use the pre-saved models}

1. Download the repository to local system.
2. Install all the dependencies as given above.
3. Open the command prompt and go to the root folder of the project.
4. Run command "python .\code\intent\CreateIntentModels.py" and give it some time.
5. Run command "python .\code\paraphrase_nlp\CleanData.py" .
6. Run command "python .\code\paraphrase_nlp\CreateParaphraseModel.py" and give it some time.
7. Run command "python .\code\ExploriaFlaskAgent.py" and give it time to start.
8. Open URL "http:\\127.0.0.1:2412\" in Chrome.


