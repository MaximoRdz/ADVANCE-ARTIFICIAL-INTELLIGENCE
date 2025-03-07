# ADVANCE-ARTIFICIAL-INTELLIGENCE
## Task 1. Vehicle Maneuver Detection and Prediction

### Replication

#### 1. Install dependencies (Python 3.10)
```
pip install -r requirements.txt
```

#### 2. Create data folder
- Unzip ManiobrasSimulador.zip
- Create a folder named 'data'
- Move DriverX folders to data folder
- **Error**: Typo in filename `Driver5/STISIMData_3step-Turning` should be `Driver5/STISIMData_3step-Turnings`

### Content
1.  [preprocessing.ipynb](https://github.com/MaximoRdz/ADVANCE-ARTIFICIAL-INTELLIGENCE/blob/main/p1/preprocessing.ipynb): notebook with the preprocessing of the original data.
2.  [experimentation.ipynb](https://github.com/MaximoRdz/ADVANCE-ARTIFICIAL-INTELLIGENCE/blob/main/p1/experimentation.ipynb): notebook with the experimentation and comparison of the models.
3.  [maneuvers.ipynb](https://github.com/MaximoRdz/ADVANCE-ARTIFICIAL-INTELLIGENCE/blob/main/p1/maneuvers.ipynb): notebook with the results and analysis of each maneuver with the best model.


## Task 2. Tissue Classification and Tumor Detection in Medical Images

### Sources
- [Dataset](https://zenodo.org/records/1214456)
- [Paper](https://www.nature.com/articles/s41591-019-0462-y)

### Replication

#### 1. Install dependencies (Python 3.10)
```
pip install -r requirements.txt
```

#### 2. Create data folder
- Create a folder named 'data'.
- Move train zip (NCT-CRC-HE-100K.zip) and test zip (CRC-VAL-HE-7K.zip) to data folder.

#### 3. Create models folder
- Create a folder named 'models' where the trained models will be saved.

### Content
1.  [tissues.ipynb](https://github.com/MaximoRdz/ADVANCE-ARTIFICIAL-INTELLIGENCE/blob/main/p2/tissues.ipynb): notebook with a multilabel (9) classifier which classifies the tissue type of the images.
2.  [tumors.ipynb](https://github.com/MaximoRdz/ADVANCE-ARTIFICIAL-INTELLIGENCE/blob/main/p2/tumors.ipynb): notebook with a binary (healthy, tumor) classifier which detects the presence of tumors in the tissue images.