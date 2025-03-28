# {{ cookiecutter.model_name }}

{{ cookiecutter.model_description }}

## Setup environment

Setup your Mlflow credentials:

- `export MLFLOW_TRACKING_TOKEN=`
- `export LOGNAME=`

## Build your docker images

- Train image: `docker build . -f Dockerfile.train -t train:latest`
- Inference image: `docker build . -f Dockerfile.inf -t inference:latest --build-arg ONNX_PATH=<path-to-onnx>`

## Install dependencies

First let's create a virtualenv:

```bash
python -m venv .venv
source .venv/bin/activate
```

### Training

```bash
pip install -r requirements-train.txt
```

### Inference

```bash
pip install -r requirements-inf.txt
```
