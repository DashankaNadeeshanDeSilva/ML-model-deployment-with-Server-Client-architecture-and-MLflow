# ML Model Deployment with Server-Client Architecture and MLflow

## Motivation and Objectives

### Motivation
The primary motivation of this project is to implement, test, and deploy a simple deep learning classifier model using a Server-Client architecture. This approach aims to demonstrate the integration of MLflow for model tracking and management, along with CI/CD pipelines for deployment.

### Objectives
1. Build a robust Server-Client architecture for ML model deployment.
2. Use MLflow for efficient experiment tracking and model management.
3. Ensure modularity and scalability for production-ready applications.
4. Integrate CI/CD pipelines for automated testing and deployment.

---

## Overview

This project demonstrates a robust setup for deploying machine learning models using a **Server-Client Architecture** with **MLflow** for model tracking and serving. The project is designed with scalability and modularity in mind, making it ideal for production-level deployment.

---

## Features

- **Server-Client Architecture**:
  - Server hosts the ML model and handles inference requests.
  - Client sends requests to the server and displays results.
- **MLflow Integration**:
  - Tracks experiments, models, and metrics.
  - Serves models via MLflow REST APIs.
- **Modular Design**:
  - Clear separation of server and client codebases.
- **CI/CD**:
  - Automated workflows using GitHub Actions and AWS deployment.

---

## Project Structure

```plaintext
ML-model-deployment-with-Server-Client-architecture-and-MLflow/
├── .git/                  # Git metadata
├── .github/               # GitHub Actions workflows
├── client/                # Client-side application
├── server/                # Server-side application
├── README.md              # Project documentation
├── requirements.txt       # Python dependencies
```

### Key Directories

- **client/**: Contains the client-side code for sending requests and visualizing results.
- **server/**: Contains server-side code, including MLflow integration and model inference logic.
- **.github/**: CI/CD pipeline configurations.

---

## Architecture

### Diagram

```plaintext
+----------------+          HTTP/REST         +----------------+
|    Client      | <------------------------> |     Server      |
| (Frontend/CLI) |                             | (Model Hosting) |
+----------------+                             +----------------+
        |                                               |
        |                MLflow Serving                |
        +----------------------------------------------+
```

### Workflow

1. **Model Training and Registration**:
   - Train the model and register it with MLflow.
   - Store model artifacts and metadata.
2. **Server Setup**:
   - Start an MLflow server for model hosting.
   - Implement a FastAPI or Flask server to handle client requests.
3. **Client Interaction**:
   - Client sends data to the server via HTTP.
   - Server processes data, performs inference, and returns results.

---

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- Docker (optional, for containerized deployment)

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd ML-model-deployment-with-Server-Client-architecture-and-MLflow
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Server

1. Navigate to the `server/` directory:
   ```bash
   cd server
   ```
2. Start the server:
   ```bash
   python app.py
   ```
3. (Optional) Use Docker to run the server:
   ```bash
   docker build -t ml-server .
   docker run -p 8000:8000 ml-server
   ```

### Running the Client

1. Navigate to the `client/` directory:
   ```bash
   cd client
   ```
2. Start the client:
   ```bash
   python app.py
   ```

### Accessing MLflow UI

1. Start the MLflow tracking server (if not running):
   ```bash
   mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000
   ```
2. Open the MLflow UI:
   - Navigate to `http://localhost:5000` in your browser.

---

## CI/CD Pipeline

The project uses **GitHub Actions** for continuous integration and deployment:

1. **Build and Test**:
   - Automatically triggered on push.
   - Runs unit tests and checks code quality.
2. **Deployment**:
   - Docker-based deployment is automated using workflows.

### Workflow Configuration

- The pipeline for Docker-based deployment is defined in `.github/workflows/deploy.yml`.
- The pipeline for AWS deployment is defined in `.github/workflows/deploy_ml_model.yml`.

---

## Future Enhancements

- Add authentication for secure client-server communication.
- Use Kubernetes for scaling.
- Implement a user-friendly frontend for the client.
- Automate model retraining and deployment pipelines.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a feature branch.
3. Submit a pull request.

---

## Contact

For questions or suggestions, please create a issue

---

Happy Deploying!

