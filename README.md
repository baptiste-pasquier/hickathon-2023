# Hickathon 2023

# Usage
## 1. Installation

1. Clone the repository
```bash
git clone https://github.com/baptiste-pasquier/m2ds_data-stream-project
```

2. Install the project
- With `poetry` ([installation](https://python-poetry.org/docs/#installation)) :
```bash
poetry install
```
- With `pip` :
```bash
pip install -e .
```

## 2. Usage
```bash
python run.py
```
```
optional arguments:  -h, --help  show this help message and exit
  --train     Train the model
  --predict   Predict on the test dataset
  --small     Use the small dataset
```
Arguments :

- `--train` : train the model
  - datasets : `/datasets/train/train_features_sent.csv` and `/datasets/train/train_labels_sent.csv`
  - output : trained model at `/trained_model/trained_model.pickle`
- `--predict` : predict with the trained model
  - dataset : `datasets/test/test_features_sent.csv`
  - output : `datasets/test/submission_sent.csv`
- `--small` : use the datasets in the `small` subfolder
