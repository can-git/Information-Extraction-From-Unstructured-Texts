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

After downloading, dataset must be in the specific format as below:

```
data
    │
    ├── data.tsv
    │
    ├── sample_submission.csv
    │
    ├── test.csv
    │
    └── train.csv
```

## Additional Details

The evaluation metric for this competition is negative log likelihood loss.
For every pathology report in the dataset, submission files should contain five columns: patient_id, likelihood_G1, likelihood_G2, 
likelihood_G3, and likelihood_G4. Please see the sample_submission.csv for the submission file format.


## Hyperparameters
Models' important parameters can be adjusted from the terminal or Main.py file as default.

```
Options:
  --name TEXT            Name of the model. (cnn8, resnet18 or densenet121)
  --batch_size INTEGER   Batch Size
  --num_workers INTEGER  Num Workers
  --epochs INTEGER       Epochs
  --lr FLOAT             Learning Rate
  --wd INTEGER           Weight Decay
  --gamma FLOAT          Gamma
  --save BOOLEAN   Save Model at the end
  --im_size INTEGER      Image Size
```

## Training

To train the model, run this command or with the desired parameters as "--name":

```train
python Main.py --name <modelname>
```
This will create a Results folder, and model will be saved by the name value in the Folder.
<br />*Currently a few models are supported(cnn8, resnet18 and densenet121)*

## Evaluation

To evaluate model with lung dataset, ".pt" file should be as in the example:

```results
Results
       │
       └── <modelname>
                      └── <modelname>_model.pt
```
If the format is as above, then the code below will work successfully.

```eval
python Evaluation.py --name <modelname>
```
This will save some metric result images in the Results folder.

## Results

Our models achieve the following performances on :


## Problems :exclamation:
There is a Data Leakage in the proposed dataset! :smiling_face_with_tear: Therefore all the results are useless.

## Contributing
This project is prepared for the ARI5004 (Deep Learning) course at Bahçeşehir University. 
Thank you to my professor Mustafa Umit Oner for all the achievements.

Dataset and idea is borrowed from [Deepslide](https://github.com/BMIRDS/deepslide), Thanks for their excellent work!
