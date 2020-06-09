# This is the file that implements a flask server to do inferences. It's the file that you will modify
# to implement the prediction for your own algorithm.

from __future__ import print_function

import os, sys, stat
import json
import shutil
import flask


from flask import Flask, jsonify
from gensim.models.doc2vec import Doc2Vec

MODEL_PATH = '/opt/ml/'
TMP_MODEL_PATH = '/tmp/ml/model'
DATA_PATH = '/tmp/data'
MODEL_NAME = '' 


if os.path.exists(MODEL_PATH):
    model_file = '/model/model.tar.gz'
    path, MODEL_NAME = os.path.split(model_file)
    #print('MODEL_NAME holds: ' + str(MODEL_NAME))
    shutil.copy(model_file, TMP_MODEL_PATH)


# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.
class ClassificationService(object):

    @classmethod
    def get_model(cls):
        """Get the model object for this instance."""
        return Doc2Vec.load(TMP_MODEL_PATH)#default model name of export.pkl

    @classmethod
    def predict(cls, input):
        """For the input, do the predictions and return them."""
        learn = cls.get_model()
        inf_input = learn.infer_vector(['run', 'codeploy', 'agent', 'user'])
        sims = learn.docvecs.most_similar([inf_input], topn=5)
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
    
    # Do the prediction
    predictions = ClassificationService.predict(img) #predict() also loads the model
    
    #print('predictions: ' + str(predictions[0]) + ', ' + str(predictions[1]))
    
    # Convert result to JSON
    # return_value = { "predictions": {} }
    # return_value["predictions"]["class"] = str(predictions[0])
    print(predictions)

    return jsonify(predictions)
