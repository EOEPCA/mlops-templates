# {{ cookiecutter.dataset_name }}

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
dvc remote modify --local workspace access_key_id '<access-key-id>'
dvc remote modify --local workspace secret_access_key '<secret-access-key>'
```
