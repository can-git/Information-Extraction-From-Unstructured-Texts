# Information Extraction From Unstructured Texts

This project was created for the Bahcesehir University ARI5005 (Deep Learning) course. Details can be accessed from the 
[Kaggle](https://www.kaggle.com/competitions/bau-ari5004-fall22-a3).


Cancer grading is important for therapeutic decision making and prognosis prediction. Pathologists grade tumors based on the morphology of cells 
within tissue and state in the pathology reports in addition to some other clinical parameters, like stage, size of the tumor, and anatomical position. 
In retrospective studies, extracting these information from unstructured text in pathology reports is crucial for sample selection and experiment design.

**Aim:** To design deep learning models extracting Fuhrman grade from pathology reports of renal cell carcinoma patients in The Cancer Genome Atlas (TCGA).

**Pathology Report Text:**
``` text
procedure: right radical nephrectomy tumor type: papillary renal cell carcinoma nuclear grade: grade iii tumor size: 
12 x 11.5 x 9 cm renal vein invasion: not present lymph nodes: not examined phthologic stage: t2 nx mx
``` 
**Fuhrman grade:**
``` text
G3
``` 

<p align="center">
<img src="etc/bg.png" height=350>
</p>

## Requirements

Python packages required (can be installed via pip or conda):

``` 
numpy
pandas
torch
nltk
scikit-learn
transformers
```

## Data Preparation

A proposed dataset can be achieved from [Kaggle](https://www.kaggle.com/competitions/bau-ari5004-fall22-a3/data).

After downloading, dataset must be inside of the "data" folder as below:

```
data
    │
    ├── data.tsv
    ...
```

## Additional Details

The evaluation metric for this competition is negative log likelihood loss.
For every pathology report in the dataset, submission files should contain five columns: patient_id, likelihood_G1, likelihood_G2, 
likelihood_G3, and likelihood_G4. Please see the sample_submission.csv for the submission file format.


## Hyperparameters
Models' important parameters can be adjusted from the Proporties.py file.

```
Options:
    -BATCH_SIZE = 4
    -EPOCHS = 20
    -LR = 0.000008
    -WD = 25e-2
    -GAMMA = 1e-1
    -STEP_SIZE = 1
    -DEVICE = "cuda"
    -SEED = 1
    -LOG_INTERVAL = 10
    -SAVE_MODEL = True
    -DEFAULT_PATH = "data"
    -DEFAULT_PATH_DATA = "data/data.tsv"
    -DEFAULT_PATH_SAMPLE = "data/sample_submission.csv"
    -DEFAULT_PATH_TEST = "data/test.csv"
    -DEFAULT_PATH_TRAIN = "data/train.csv"
    -BERT_NAME = "dmis-lab/biobert-v1.1"
```

## Train and Evaluating

To train the model, the following code in the Main.py file should be as shown:

```python
train(model, df_train, df_val, criterion, optimizer)

# model.load_state_dict(torch.load('model.pt'))
# evaluate(model, df_test)
```

To evaluate the model:

```python
# train(model, df_train, df_val, criterion, optimizer)

model.load_state_dict(torch.load('model.pt'))
evaluate(model, df_test)
```

The evaluate function creates a csv file as requested on the [Kaggle](https://www.kaggle.com/competitions/bau-ari5004-fall22-a3/overview/evaluation) website.

## Results

The required CSV file can be accessed from the "results/" folder. The result is a negative log likelihood score of 0.16081. The leaderboard on the Kaggle site is as follows:
[Kaggle for Course ARI5004](https://www.kaggle.com/competitions/bau-ari5004-fall22-a3/leaderboard),
[Kaggle for Course AIN2001](https://www.kaggle.com/competitions/bau-ain2001-fall22-a4/leaderboard),

## Contributing
This project is prepared for the ARI5004 (Deep Learning) course at Bahçeşehir University. 
Thank you to my professor Mustafa Umit Oner for all the achievements.
