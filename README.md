# Midterm Project, Deep Learning

Two separate deep learning studies. Each one has a problem definition, a full training
pipeline, a simpler control model for comparison, and an evaluation on held out test data.

| # | Model | Dataset | Task | Test result |
|---|-------|---------|------|-------------|
| 1 | DNN on tabular data | [UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing), 45,211 rows, 51 encoded features | Predict whether a client subscribes to a term deposit | ROC-AUC **0.9240**, F1 **0.618** at a tuned threshold (logistic regression: ROC-AUC 0.9047, F1 0.548) |
| 2 | CNN on images | [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist), 70,000 greyscale images of 28x28 pixels, 10 classes | Assign a product category to each garment image | Accuracy **92.64 %**, macro-F1 **0.9264** (pixel MLP: 86.74 %) |

## Files

| File | What it is |
|------|------------|
| `report/Project_Report.pdf` | The project document (28 pages), with the required sections for both models |
| `report/Project_Report.docx` | Same document, editable. Fill in name / roll number on the title page |
| `notebooks/DNN_Bank_Marketing.ipynb` | Executed notebook for Model 1: code, outputs, figures |
| `notebooks/CNN_Fashion_MNIST.ipynb` | Executed notebook for Model 2: code, outputs, figures |
| `src/dnn_bank_marketing.py` | Script for Model 1: EDA, preprocessing, training, threshold tuning, evaluation |
| `src/cnn_fashion_mnist.py` | Script for Model 2: augmentation, CNN training, evaluation, error analysis |
| `src/download_data.py` | Downloads both datasets into `data/` |
| `src/build_notebooks.py` | Builds both notebooks from the `src/` code and executes them |
| `report/make_report.py` | Builds the DOCX and PDF from `results/*/metrics.json` |

## Report outline

Both models are written up with the same sections:

Problem Definition, Dataset Description and Source Link, Proposed Solution, Data
Preprocessing, Model Architecture, Training Setup, Training Results and Evaluation,
Discussion. After the two model chapters there is a comparison of the two studies, a
conclusion, references, and two appendices (project structure and output files).

## Re-running everything

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python src/download_data.py          # downloads both datasets into data/, about 1 min
python src/dnn_bank_marketing.py     # Model 1, about 1 min
python src/cnn_fashion_mnist.py      # Model 2, about 2 min on a GPU
python src/build_notebooks.py        # rebuilds and executes both notebooks
python report/make_report.py         # rebuilds the report from the new metrics
```

`build_notebooks.py` needs a Jupyter kernel called `dl-project`. Register it once:

```bash
python -m ipykernel install --user --name dl-project --display-name "Python (DL project)"
```

The PDF is produced with `libreoffice --headless --convert-to pdf`. If LibreOffice is not
installed the DOCX is still written, and it can be exported to PDF by hand.

## Output folders

```
results/dnn/   01_class_balance_and_age.png  02_categorical_effects.png  03_training_curves.png
               04_confusion_matrix.png  05_roc_pr_curves.png  06_threshold_analysis.png
               07_permutation_importance.png  metrics.json  history.json  model_dnn.pt
               preprocessor.json  permutation_importance.csv
results/cnn/   01_sample_images.png  02_class_distribution.png  03_training_curves.png
               04_confusion_matrix.png  05_per_class_metrics.png  06_predictions.png
               07_misclassified.png  metrics.json  history.json  model_cnn.pt
```

When a notebook is executed it writes its figures to `results/notebook/dnn/` and
`results/notebook/cnn/`, so the figures produced by the scripts are not overwritten.

The report reads `metrics.json` directly, so every number in the document comes from the
last run of the scripts.

## Notes on the two models

**Model 1.** Stratified 70/15/15 split (31,647 / 6,782 / 6,782 rows). The scaler and the
one hot encoding are fitted on the training split only. The class imbalance (11.7 %
positive) is handled with `pos_weight`, and the decision threshold is tuned on the
validation split for F1 and then applied once to the test set. Two controls are reported:
logistic regression on the same features, and an ablation that removes `duration`. The
duration column is only known after the call has finished, so it cannot be used for
targeting; without it ROC-AUC falls from 0.924 to 0.781, and the duration free model is
the one that could actually be deployed.

**Model 2.** Three VGG style blocks (two 3x3 convolutions each, then BatchNorm, ReLU and
max pooling), dropout 0.25 and 0.4, global average pooling, label smoothing 0.1, AdamW with
cosine annealing, 15 epochs with early stopping on validation accuracy (best epoch 14,
about 58 s on an RTX 5070). The control is an MLP on flattened pixels. It has 535,818
parameters against 288,170 for the CNN and is still 5.9 points less accurate, which shows
that the gain comes from the convolutional structure and not from model size.

## Environment

Python 3.10+ (this run used 3.12.3), PyTorch 2.9.1, CUDA GPU optional. All seeds are fixed
to 42 and DataLoader shuffling is seeded, so a re-run on the same hardware gives the same
numbers.
