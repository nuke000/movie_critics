from flask import Flask, request
import json
import pickle
import re
import requests 
from SentimentClassifier import SentimentClassifier
import time


app = Flask(__name__)

model = SentimentClassifier()
model.fit_model()

@app.route('/')
def help_message():
    message = "Use POST to /sentiment path to classify the sentiment of the movie review.\n" 
    message += "Post json request format {'review': 'text of review for classification'}.\n" 
    message += "Expect json reply format {'sentiment': 1} for a positive review, {'sentiment': 0} for a negative review.\n" 
    message += ("Current model accuracy: " + str(model._accuracy))
    return message

@app.route('/sentiment', methods=["GET", "POST"])
def sentim_classifier():
    if request.method == 'POST':
        rq = request.get_json()
        review = rq['review']
        result = model.predict(review)
        if result == 1 :
            response = {"sentiment": 1}
        else :
            response = {"sentiment": 0}
        return json.dumps(response)
    else:
        message = "You should use only POST query to classify the sentiment of the movie review.\n" 
        message += "POST json request format {'review': 'text of review for classification'}.\n" 
        message += "Expect json reply format {'sentiment': 1} for a positive review, {'sentiment': 0} for a negative review.\n" 
        return message

if __name__ == '__main__':
    app.run("0.0.0.0", 8000)
