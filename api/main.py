from fastapi import FastAPI,File,UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import requests


app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CLASS_NAMES = ['Early_blight', 'Late_blight', 'Healthy']
# latest_model = os.listdir("./models")
# print(latest_model)
PATH = "./models/1.keras"
MODEL = tf.keras.models.load_model(PATH)
# MODEL = tf.keras.layers.TFSMLayer(filepath=PATH, call_endpoint='serving_default')

@app.get("/")
def home():
    return {"message":"Welcome to Potato Disease Detection"}

def read_file(data) ->np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict( file: UploadFile = File(...) ):
	print(file.filename)
	image = read_file(await file.read())
	image_batch = np.expand_dims(image, axis=0)
	predictions = MODEL(image_batch)
	print(predictions)
	prediction = CLASS_NAMES[np.argmax(predictions[0])];
	confidence = float(np.max(predictions[0]))
	return {"message":"Prediction Successful",
          		"class":prediction,
            	"confidence":confidence,
            }

if __name__ == "__main__":
	uvicorn.run(app, host='localhost',port=1242)
