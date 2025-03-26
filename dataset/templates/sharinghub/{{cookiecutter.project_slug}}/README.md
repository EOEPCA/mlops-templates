# {{ cookiecutter.dataset_name }} Dataset

{{ cookiecutter.dataset_description }}

## Getting started

### Prerequisites

Install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Authenticate DVC

Configure your authentication (will be only stored locally):

```bash
dvc remote modify --local sharinghub password '<gitlab-access-token>'
```
