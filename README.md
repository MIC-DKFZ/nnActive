# nnActive Playground

Scripts for nnActive development

Install with
```bash
pip install -e '.[dev]'
```

## Set up nnActive
1. set up nnUNetv2
2. export nnActive_results=`path...`

## Contributing

- Always run `black` (and ideally `isort`) before commiting
- Turn on `pylint` in your editor, if it shows errors:
    1. Fix the error
    2. If it is a false positive or if you have a good reason to disagree in
       this instance add `# pylint: disable=<msg>` or `# pylint: disable-next=<msg>`
       (see [message control](https://pylint.readthedocs.io/en/latest/user_guide/messages/message_control.html) and [list of checkers](https://pylint.readthedocs.io/en/latest/user_guide/checkers/features.html))
    3. If you think this error should never be reported add it to `pyproject.toml`
        ```toml
        [tool.pylint]
        disable = [
            <msg 1>,
            <msg 2>,
            ...
        ]
        ```

## Active Learning Integration

The annotated data for each loop is saved in the `loop_XXX.json` file situated in $nnUNet_raw.
These files are used for creating the validation splits for training.
It is structured as follows:
```json
{
    "patches": [
        {
            "file": "hippocampus_361.nii.gz",
            "coords": [
                0,
                0,
                0
            ],
            "size": "whole"
        },
        {
            "file": "hippocampus_230.nii.gz",
            "coords": [
                0,
                0,
                0
            ],
            "size": "whole"
        },...
    ]
    "metafile" : "xxx",
    ...
}
```
`"patches"` is used to save the annotated areas and in `loop_XXX.json` only the newly annotated areas are saved.
To recreate the dataset for `loop_002.json` needs to be aggregated with `loop_001.json` and `loop_000.json`.


## Active Learning Setup
### Prepare Source Dataset (Fully Annotated)
Create Raw Data:
```bash
nnUNetv2_convert_MSD_dataset -i {Path-to}/Task04_Hippocampus
```

1. Create Validation Split
```bash
python scripts/create_val_split.py -d 4
```
Creates folders `imagesVal` and `labelsVal` while taking some images out of the `imagesTr` and `labelsTr` folder.

3. Obtain nnU-Net preprocessing instructions
```bash
nnUNetv2_extract_fingerprint -d 4
nnUNetv2_plan_experiment -d 4
```
4. Resample images
```bash
python nnactive/resample_dataset.py --target_preprocessed ${nnUNet_preprocessed}/Dataset004_Hippocampus --target_raw ${nnUNet_raw}/Dataset004_Hippocampus
```
Alternatively:
```bash
python scripts/resample_nnunet_dataset -d 4
```
resamples images in imagesTr and labelsTr to target space. Original images are saved in `imagesTr_original` and `labelsTr_original`
Creates Folders imagesVal and labelsVal while taking some images out of the imagesTr and labelsTr folder.
### Create Partially annotated dataset
5. Create Dataset
```bash
python scripts/convert_to_partannotated.py -d 4
```
Creates dataset with offset of 500. In this case dataset 504.
Creates: 
    1. `${nnUNet_raw}/Dataset504_Hippocampus-partanno` folder structure
    2. `${nnUNet_preprocessed}/Dataset504_Hippocampus-partanno/splits_final.json`


6. Create Plans
```bash
nnUNetv2_extract_fingerprint -d 504
nnUNetv2_plan_experiment -d 504 -c 3d_fullres

# Alternatively:
nnUNetv2_plan_and_preprocess -d 504 -c 3d_fullres -np 4
```

7. Create Config and Set Up AL experiment folder
    - [ ] TODO: make this generalizable
```bash
python scripts/setup_al_experiment.py -d 504
```

## Active Learning Workflow
### Training Step
Plan & Preprocess
```bash
nnUNetv2_preprocess -d 504 -c 3d_fullres -np 4
```
#### Manual
for each fold X in (0, 1, 2, 3, 4):
```bash
nnUNetv2_train 504 3d_fullres X -tr nnUNetDebugTrainer 
```
#### nnUNet
Alternative:
```bash
python scripts/train_nnUNet_ensemble.py -d 504
```

### Prediction on external Validation/Test Set
#### nnUNet
```bash
python scripts/get_performance.py -d 504
```
Uses ensemble to compute final performance on `imagesVal` and `labelsVal` saving them in `${nnActive_results}/{dataset_name}/loop_XXX/summary.json`.

### Pool Prediction Step
#### Manual
for each fold X in (0, 1, 2, 3, 4):
```bash
nnUNetv2_predict -d 504 -c 3d_fullres -i ${nnUNet_raw}/Dataset504_Hippocampus-partanno/imagesTr -o ${nnUNet_results}/Dataset504_Hippocampus-partanno/predTr/fold_X -tr nnUNetDebugTrainer --save_probabilities -f X
```
#### nnUNet
```bash
python scripts/predict_nnUNet_ensemble.py -d 504
```


### Query Step
#### Manual
Then calculate uncertainties from softmax outputs
```bash
python nnactive/calculate_uncertainties.py -p ${nnUNet_results}/Dataset504_Hippocampus-partanno/predTr
```
Uncertainties are now in folder: `${nnUNet_results}/Dataset504_Hippocampus-partanno/predTr/uncertainties` 

These uncertainties are now aggregated:
```bash
python nnactive/uncertainty_aggregation/aggregate_uncertainties.py -i ${nnUNet_results}/Dataset504_Hippocampus-partanno/predTr/uncertainties -o  ${nnUNet_results}/Dataset504_Hippocampus-partanno/predTr/uncertainties_aggregated -d ${nnUNet_raw}/Dataset504_Hippocampus-partanno/dataset.json 
```
These uncertainties are now aggregated in the folder: `${nnUNet_results}/Dataset504_Hippocampus-partanno/predTr/uncertainties_aggregated`

These are used to query patches for loop X:
```bash
python nnactive/query_patches.py -i ${nnUNet_results}/Dataset504_Hippocampus-partanno/predTr/uncertainties_aggregated -u mutual_information -n 20 -o ${nnUNet_raw}/Dataset504_Hippocampus-partanno -l X
```
#### nnUNet
```bash
python scripts/query_step.py -d 504
```

#### Outcome
Then two files are created in `${nnUNet_raw}/Dataset504_Hippocampus-partanno`:
1. `loop_00X.json`
2. `{uncertainty_type}_loop00X.json`

### Update Dataset
#### Manual
Now the dataset needs to be updated according to all `loop_XXX.json` files:
```bash
python nnactive/update_data.py -i ${nnUNet_raw}/Dataset004_Hippocampus -p ${nnUNet_raw}/Dataset504_Hippocampus-partanno --save_splits_file ${nnUNet_preprocessed}/Dataset504_Hippocampus-partanno/splits_final.json
```

#### nnUNet
```bash
python scripts/update_data.py -d 504
```

Start with Training of all folds and repeat....



## Requirements
dataset.json in raw data
```json
{
    "channel_names": {
        "0": "MRI"
    },
    "description": "Left and right hippocampus segmentation",
    "file_ending": ".nii.gz",
    "labels": {
        "Anterior": 1,
        "Posterior": 2,
        "background": 0,
        "ignore": 3
    },
    "licence": "CC-BY-SA 4.0",
    "name": "Hippocampus-partanno",
    "numTest": 130,
    "numTraining": 260,
    "reference": " Vanderbilt University Medical Center",
    "relase": "1.0 04/05/2018",
    "tensorImageSize": "3D",
    "annotated_id" : 4 
    // id to annotated dataset or path?
}
```