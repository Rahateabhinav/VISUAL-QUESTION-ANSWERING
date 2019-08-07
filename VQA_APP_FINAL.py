from flask import Flask, render_template, request,url_for
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tablib
import uuid
import os

import json
import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize #For word tokenization
from nltk.corpus import stopwords # For removing stop words
from nltk.stem import WordNetLemmatizer # For Lemmentizing
#from spellchecker import SpellChecker  # For Spell Correction
from nltk.corpus import wordnet
import spacy
import string
import os
from sklearn.preprocessing import LabelEncoder
from keras.applications.vgg16 import VGG16
import cv2
import numpy as np
from keras.models import model_from_json
from string import punctuation


from keras.applications.vgg16 import preprocess_input
import os
from keras.preprocessing.image import img_to_array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.image import img_to_array
from tensorflow.python.keras.models import load_model
from keras.models import load_model as load_model2
import joblib
import keras

app = Flask(__name__, template_folder='template', static_folder='static')
#VGG_model=load_model("VGG_model_test_final.h5")
#VQA_model=load_model2("VQA_FINAL_model.h5")
app_root = os.path.dirname(os.path.abspath(__file__))


@app.route('/')
def index():
    return render_template('DEMO.html')

@app.route('/upload',methods=['GET', 'POST'])
def img_submit():
    target = os.path.join(app_root, 'static/')
    #print(target)

    if not os.path.isdir(target):
        os.mkdir(target)

    if request.method == 'POST':
        global destination
        _image = request.files["image"]
        #print(_image)
        #filename = _image.filename
        res = uuid.uuid4().hex
        global imagename
        imagename = '{}.jpg'.format(res)
        destination = "/".join([target, imagename])
        _image.save(destination)

    return render_template('DEMO.html',image=imagename)

@app.route('/go',methods=['GET', 'POST'])
def question_read():
    if request.method == 'POST':
        global _question
        _question = request.form['question']
        #print(_question)


    def word_preprocess(question):
    # Converting it into Lower String
        question = question.lower()

    # Removing punctuations
        question = ''.join(c for c in question if c not in punctuation)

    # Removing Stop words
        stop_words = set(stopwords.words('english'))
        question = " ".join(x for x in question.split() if x not in stop_words)

        return question

    def Tokenize(question):
        t = Tokenizer()
        t.fit_on_texts(question)
        vocab_size = len(t.word_index) + 1
    # integer encode the documents
        question = t.texts_to_sequences(question)
        return question

    def padding(token):
        from keras.preprocessing.sequence import pad_sequences as keras_pad_sequences
        token = pad_sequences(token,maxlen=26,padding = "pre",truncating="post",value = 0)
        return token

    def img_pre(imag):
        image = img_to_array(imag)
        img_resized=cv2.resize(image, (224,224), interpolation = cv2.INTER_AREA)
        image = img_resized.reshape((1,img_resized.shape[0], img_resized.shape[1], img_resized.shape[2]))
        return image

    def load_vgg(model=None):
        model = load_model("VGG_model_test_final.h5")
        return model

    def load_vqa(model2=None):
        model2 = load_model2("VQA_FINAL_model.h5")
        return model2

    def testing(img, ques):
        word = word_preprocess(ques)
        token = Tokenize(word)
        pad = padding(token)
        image = img_pre(img)
        #print(image.shape)
        VGG_model = load_vgg()
        features = VGG_model.predict(image)
        #print(features.shape)
        #print(pad)
        return features, pad

    def testing2(img, ques):
        features, pad = testing(img, ques)
        global ans
        VQA_model = load_vqa()
        ans = VQA_model.predict([features, pad])
        keras.backend.tensorflow_backend.clear_session()
        return ans

    path = "/static /imagename"
    imag = cv2.imread(destination)
    #print(imag.shape)
    y_output = testing2(imag, _question)

    labelencoder = joblib.load('labelencode_train.pkl')
    x = np.argsort(y_output)[0, -5:]
    lab = labelencoder.inverse_transform(x)
    lst = []
    for i, j in zip(x[::-1], lab[::-1]):
        lst.append((str(y_output[0, i] * 100).zfill(5)+ "% "+ j))

    return render_template('DEMO.html',ans1=lst[0],ans2=lst[1],ans3=lst[2],ans4=lst[3],ans5=lst[4],image=imagename)

if __name__ == '__main__':
    app.run(debug=True)
