from flask import Flask
import joblib

app = Flask(__name__)

model = joblib.load('model/model_binary.dat.gz')
