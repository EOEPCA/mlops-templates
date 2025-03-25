# ML Model Templates

## Description

This repository provides a structured template for developing **machine learning models**.

It includes everything needed for **data preprocessing, training, inference, and deployment** in a containerized environment.

### PyTorch

Below is an overview of the project's structure, along with the purpose of each folder and file.

#### `checkpoints/`

Stores trained model weights and states.

Useful for resuming training or running inference with a pre-trained model.

#### `datasets/`

Contains the datasets used for training and evaluation.

- You can organize it into subfolders.
- The dataset is loaded using **Hugging Face's `datasets` library**.

#### `notebooks/`

Jupyter notebooks for **experimentation, visualization, and demonstrations**.

Ideal for quick prototyping and interactive analysis.

#### `src/`

Contains the **source code** of the project, including:

- **Training scripts** (e.g., `train.py`)
- **Inference scripts** (e.g., `inference.py`)
- Additional features such as **preprocessing, augmentation, utilities**, etc.

#### `workflows/`

Holds **workflow definitions** in **CWL (Common Workflow Language)** format.

Used for defining **OGC Application Packages** to automate and standardize model execution.

#### `Dockerfile.*`

Defines Docker images for running the project in a **containerized environment**.

Separate Dockerfiles can be used for **training, inference, and deployment**.

#### `pyproject.toml`

Specifies **project dependencies** and package configurations.

Used with **Poetry** for dependency management.

#### `requirements*.txt`

Lists dependencies required for **Docker images**.

For inference, it is recommended to use **ONNX Runtime** when exporting the model in `.onnx` format.
