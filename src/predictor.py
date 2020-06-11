# This is the file that implements a flask server to do inferences.
import gensim
import flask
import json
import glob
from gensim.models.doc2vec import Doc2Vec

# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.
class ClassificationService(object):

    @classmethod
    def get_model(cls):
        """Get the model object for this instance."""
        modelfile = glob.glob('/opt/ml/model/*.pkl')[0]
        return Doc2Vec.load(modelfile)

    @classmethod
    def preprocess_text(cls, test_input):
        tokens = gensim.utils.simple_preprocess(test_input)
        return tokens

    @classmethod
    def predict(cls, input):
        """For the input, do the predictions and return them."""
        learn = cls.get_model()
        print('input is ', input)
        tokens = cls.preprocess_text(input)
        print('TOKENS ARE ', tokens)
        inf_input = learn.infer_vector(tokens)
        sims = learn.docvecs.most_similar([inf_input], topn=5)
        print('most similar docs and its similarity', sims[0])
        return sims[0][0]

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
    print('Request payload ', payload)
    # Do the prediction
    predictions = ClassificationService.predict(payload) #predict() also loads the model
    print('Most similar doc ID is ', predictions)
    print('Now constructing JSON out of this prediction ......')
    output = {}
    output['QuestionID'] = str(predictions)
    json_output = json.dumps(output)
    return json_output