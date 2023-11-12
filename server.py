from flask import Flask, request
import json
import pickle
import re
from SentimentClassifier import SentimentClassifier

app = Flask(__name__)

model = SentimentClassifier()
model.fit_model()

@app.route('/')
def help_message():
    message = "Use /sentiment path to classify the sentiment of the movie review.\n" 
    message += "Post json format {'review': 'text of review for classification'}.\n" 
    message += (" Current model accuracy: " + str(model._accuracy))
    return message

@app.route('/sentiment', methods=["GET", "POST"])
def sentim_classifier():
    if request.method == 'POST':
        rq = request.get_json(force=True)
        review = rq['review']
        result = rewiew
        #result = model.predict(review)
        response = {
            "result": result
        }
        return json.dumps(response)
    else:
        return "You should use only POST query"

if __name__ == '__main__':
    app.run("0.0.0.0", 8000)
