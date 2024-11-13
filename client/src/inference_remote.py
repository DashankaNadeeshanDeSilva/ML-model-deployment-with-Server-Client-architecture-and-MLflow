import requests
import json
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

def load_image(image_path):
    """Load and preprocess an image for inference."""
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Resize to match model input size
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).float()  # Add batch dimension
    return image


def post_process_predictions(predictions):
    """Post process the model response from the server to get the class"""
    # get the logits from the 1st sample (For batch predictions change)
    logits = predictions["predictions"][0]
    # Softmax function across the class dimesion
    softmax = torch.nn.Softmax(dim=0)
    # get probabilities from logits with softmax also transform list of logts to tensor
    prediction_probs = softmax(torch.tensor(logits)) 
    # get class with max prob
    predicted_class = torch.argmax(prediction_probs).item() # .item due to tensor

    return predicted_class, prediction_probs.tolist()

def send_inference_request(image_tensor, server_url):
    """Send an inference request to the MLflow model server."""
    # Convert the tensor to a list for JSON serialization as Mlflow model server expects data in JSON
    input_data = image_tensor.type(torch.float32).numpy().astype(np.float32).tolist()
    #input_data = input_data.numpy().astype(np.float32).tolist()
    
    ## Prepare the payload (serialized into a JSON payload)
    payload = {
        "instances": input_data  
    }
    
    ## Send the request
    headers = {"Content-Type": "application/json"}
    """Send the HTTP POST Request"""
    response = requests.post(server_url, data=json.dumps(payload), headers=headers)

    ## Handle the Server Response: parse the response
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Request failed with status code {response.status_code}: {response.text}")

if __name__ == "__main__":
    ## Inputs ##
    # Specify the image path
    image_path = "resources/cat.jpg"
    # Specify server URL
    server_url= <MODEL_SERVER_URI>
    
    # Load and preprocess the image
    image_tensor = load_image(image_path)
    
    # Send the request to the MLflow model server and get prediction to the model
    try:
        raw_prediction = send_inference_request(image_tensor, server_url)
        predicted_class, prediction_probs_list = post_process_predictions(raw_prediction)
        # CIFAR-10 class dict
        classes_dict = {}
        classes_list = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        for i, cls in enumerate(classes_list):
            classes_dict[i] = cls

        print("Predicted Class:", classes_dict[int(predicted_class)])
        print(f"Prediction Probability: {prediction_probs_list[int(predicted_class)]*100:.2f}%")

    except Exception as e:
        print("Error during inference:", e)
