# train.py
import torch
import torch.optim as optim
import torch.nn as nn
import mlflow
import mlflow.pytorch

from model import CNN_Classifier
from data_loader import get_loaders
from utils import set_seed, initialize_weights, adjust_learning_rate, save_model

# Configuration parameters, better to move them to a config.py file
config = {
    "batch_size": 32,
    "epochs": 10,
    "learning_rate": 0.001,
    "seed": 100
}

def main():
    set_seed(config["seed"])
    train_loader, test_loader = get_loaders(config["batch_size"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = CNN_Classifier().to(device)
    model.apply(initialize_weights)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    # Set up MLflow
    mlflow.set_experiment(config["experiment_name"])
    with mlflow.start_run():
        ## Log hyper-parameter to MLflow ##
        mlflow.log_params(config) 

        for epoch in range(config["epochs"]):
            model.train()
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data[0].to(device), data[1].to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                '''if i % 200 == 199:
                    print(f'Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 200:.4f}')
                    running_loss = 0.0'''

            # Calculate loss after every epoch
            avg_loss = running_loss/ len(train_loader)
            ## Log eval metrics to MLflow ##
            mlflow.log_metric("loss", avg_loss, step=epoch)

            adjust_learning_rate(optimizer, epoch, init_lr=config["learning_rate"])

            # Evaluation
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for data in test_loader:
                    images, labels = data[0].to(device), data[1].to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total
            mlflow.log_metric("accuracy", accuracy, step=epoch)
            print(f'Epoch {epoch+1}: Loss: {avg_loss:.2f}, Accuracy: {accuracy:.2f}%')

        #save_model(model)
        ## Log the model to MLflow ##
        mlflow.pytorch.log_model(model, "model")


if __name__ == "__main__":
    main()
