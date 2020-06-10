# This is the file that implements a flask server to do inferences. It's the file that you will modify
# to implement the prediction for your own algorithm.

import os, sys, stat
import json
import shutil
import flask


from flask import Flask, jsonify
from gensim.models.doc2vec import Doc2Vec

import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
import string


MODEL_PATH = '/opt/ml/'
# TMP_MODEL_PATH = '/tmp/ml/model'
DATA_PATH = '/tmp/data'
MODEL_NAME = ''

if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH, exist_ok=True)

elif os.path.exists(MODEL_PATH):
    model_file = '../model/model.tar.gz'
    path, MODEL_NAME = os.path.split(model_file)
    shutil.copy(model_file, MODEL_PATH)


# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.
class ClassificationService(object):

    @classmethod
    def get_model(cls):
        """Get the model object for this instance."""
        return Doc2Vec.load(MODEL_PATH)#default model name of export.pkl

    @classmethod
    def preprocess_text(cls, test_input):
        test_input = test_input.lower()
        stopset = stopwords.words('english') + list(string.punctuation)
        test_input = " ".join([i for i in word_tokenize(test_input) if i not in stopset])
        return word_tokenize(test_input)

    @classmethod
    def predict(cls, input):
        """For the input, do the predictions and return them."""
        learn = cls.get_model()
        print('input is ', input)
        tokens = cls.preprocess_text(input)
        print('TOKENS ARE ', tokens)
        inf_input = learn.infer_vector(tokens)
        sims = learn.docvecs.most_similar([inf_input], topn=5)
        print('most similar docs ', sims[0])
        return sims[0]

# The flask app for serving predictions
app = flask.Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = ClassificationService.get_model() is not None  

    status = 200 if health else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def transformation():

    payload = flask.request.data
    print('payload ', payload)
    # Do the prediction
    predictions = ClassificationService.predict(payload) #predict() also loads the model
    print(predictions)

    return jsonify(predictions)
