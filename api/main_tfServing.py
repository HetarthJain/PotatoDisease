from fastapi import FastAPI,File,UploadFile
import uvicorn
import os
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import requests
import json

app = FastAPI()

endpoint = "http://localhost:1244/v1/models/saved_model/predict"

# prod = "http://localhost:1244/v1/models/saved_model:predict"
# beta = "http://localhost:1244/v1/models/potatoes_model/versions/2:predict"
headers = {"content-type": "application/json"}
CLASS_NAMES = ['Early_blight', 'Late_blight', 'Healthy']

@app.get("/")
def home():
    return {"message":"Welcome to Potato Disease Detection"}

def read_file(data) ->np.ndarray:
    print("reading image")
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
    ):
		image = read_file(await file.read())
		image_batch = np.expand_dims(image, axis=0).tolist()
		data = {
			"instances":image_batch}
		
		requests.get(endpoint,{"message":"getting result"})
		response = requests.post(endpoint,data=json.dumps(data),headers=headers)
		print(response.json())
		# predictions = np.array(response.json()["predictions"][0])

		# prediction = CLASS_NAMES[np.argmax(predictions)]
		# confidence = np.max(prediction)
		# return {"message":"Prediction Successful",
        #   		"class":prediction,
        #     	"confidence":confidence,
        #     }

if __name__ == "__main__":
	uvicorn.run(app, host='localhost',port=1243)