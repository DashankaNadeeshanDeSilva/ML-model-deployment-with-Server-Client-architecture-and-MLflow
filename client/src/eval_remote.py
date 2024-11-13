
import torch
import argparse
import requests
import json
import numpy as np

from dotenv import load_dotenv
import os

# import data loader utility
from data_loader import get_loaders

load_dotenv()

def inference_api_call(input_data, server_uri):
    """Send an inference request to the MLflow model server."""
    # Convert the tensor to a list for JSON serialization as Mlflow model server expects data in JSON
    input_data = input_data.type(torch.float32).numpy().astype(np.float32).tolist()
    
    ## Prepare the payload (serialized into a JSON payload)
    payload = {
        "instances": input_data  
    }
    
    ## Send the request
    headers = {"Content-Type": "application/json"}
    """Send the HTTP POST Request"""
    response = requests.post(server_uri, data=json.dumps(payload), headers=headers)

    ## Handle the Server Response: parse the response
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Request failed with status code {response.status_code}: {response.text}")


def evaluate_model(mlflow_model_server_uri):

	# Load the test dataset
	test_loader = get_loaders(batch_size=256)

    # init metrics
	correct_pred = 0
	total = 0

    # eval loop
	with torch.no_grad():
		for images, labels in test_loader:
			outputs = inference_api_call(images, mlflow_model_server_uri)

			outputs = torch.tensor(outputs["predictions"])
			outputs = torch.argmax(outputs, dim=1)

			total += labels.size(0)
			correct_pred += (outputs == labels).sum().item()

	accuracy = 100 * (correct_pred/total)

	print('Accuracy: ', accuracy)

if __name__ == '__main__':

	mlflow_model_server_uri=os.getenv("MODEL_SERVER_URI")
	evaluate_model(mlflow_model_server_uri)




