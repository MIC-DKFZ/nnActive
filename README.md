# nnActive Playground

Scripts for nnActive development

Install with
```bash
# use Pytorch 2.4.0 and CUDA 12.4
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -e '.[dev]'
```

We recommend a swap memory size of ≥128GB to avoid OOM issues.

## Set up nnActive
Set up paths as follows:
```bash
export nnActive_raw=Path_to_raw_nnunet_data # contains base datasets are derived
export nnActive_data=Path_to_nnactive_data # contains data for Active Learning experiments
export nnActive_results=Path_to_nnactive_results # contains results from Active Learning experiments
export nnUNet_raw=$nnActive_raw/nnUNet_raw # base_datasets (ID)
export nnUNet_preprocessed=$nnActive_raw/nnUNet_preprocessed # base datasets (ID)
export nnUNet_results=Path_to_nnUnet_results # base datasets (ID)
```
### Autocompletion

#### Usage:
```bash
nnactive setup --experiment <tab>
nnactive run_al_loops --experiment <tab>
```
#### Set up on ZSH:
write completion file
```bash
nnactive -s zsh > $HOME/.local/bin/completions/_nnactive
```
Add to fpath
```
fpath+="$HOME/.local/bin/completions"
```
#### Set up on Oh-my-zsh:
write completion file
```bash
nnactive -s zsh > $HOME/.oh-my-zsh/completions/_nnactive
```

#### Add a new function to CLI
Add import to file which carries function in: `nnactive/cli/__init__.py`

### Setting up the data
Each dataset needs to be in the standard nnU-Netv2 format.
E.g. do so with the Medical Segmentation Decathlon Hippocampus Dataset
Create Raw Data:
```bash
nnUNetv2_convert_MSD_dataset -i {Path-to}/Task04_Hippocampus
```
Inside of these paths the following files 
```bash
$nnUNet_raw
├── Dataset004_Hippocampus
│   ├── dataset.json
│   ├── imagesTr
│   └── labelsTr
...
```

Now create the seperation into validation and training set for Active Learning.
```bash
nnactive init_create_val_split --dataset_id 4 
```
```bash
$nnUNet_raw
├── Dataset004_Hippocampus
│   ├── dataset.json
│   ├── imagesTr
│   ├── imagesVal
│   ├── labelsTr
│   └── labelsVal
...
```

Now set everything into the spacing used for the nnU-Net configuration.
```bash
nnUNetv2_plan_and_preprocess -d 4 -c 3d_fullres -np 8
nnactive init_resample_from_id --dataset_id 4 
```
```bash
$nnUNet_raw
├── Dataset004_Hippocampus
│   ├── dataset.json
│   ├── imagesTr #resampled data
│   ├── imagesTr_original 
│   ├── imagesVal #resampled data
│   ├── imagesVal_original
│   ├── labelsTr #resampled data
│   ├── labelsTr_original
│   ├── labelsVal #resampled data
│   └── labelsVal_original
...
```

Creation of small derivative datasets
```bash
nnactive nnactive init_create_small_dataset ----base_dataset_id 4 --target_dataset_id 999
```
```bash
$nnUNet_raw
├── Dataset004_Hippocampus
│   ├── dataset.json
│   ├── imagesTr #resampled data
│   ├── imagesTr_original 
│   ├── imagesVal #resampled data
│   ├── imagesVal_original
│   ├── labelsTr #resampled data
│   ├── labelsTr_original
│   ├── labelsVal #resampled data
│   └── labelsVal_original
├── Dataset999_Hippocampus_small
│   ├── dataset.json
│   ├── imagesTr #resampled data
│   ├── imagesVal #resampled data
│   ├── labelsTr #resampled data
│   └── labelsVal #resampled data
...
```


## Active Learning Experiment
After the base dataset has been set up we create an experiment with the setup function
```bash
nnactive setup_experiment --experiment Hippocampus__patch-20_20_20__qs-40__unc-random__seed-12345
```
This creates the following folders:
```bash
$nnActive_data
├── Dataset004_Hippocampus # base_dataset folder
│   ├── nnUNet_preprocessed
│   │   ├── Dataset000_Hippocampus__patch-20__qs20__unc-random__seed-12345
│   │   │   ├── nnUNetPlans.json
│   │   │   ├── gt_segmentations
│   │   │   └── nnUNetPlans_3d_fullres
│   ├── nnUNet_raw
│   │   ├── Dataset000_Hippocampus__patch-20__qs20__unc-random__seed-12345
│   │   │   ├── loop_000.json # contains annotated patches
│   │   │   ├── imagesTr -> $nnActive_raw/nnUNet_raw/Dataset004_Hippocampus/imagesTr
│   │   │   ├── imagesTs -> $nnActive_raw/nnUNet_raw/Dataset004_Hippocampus/imagesTs
│   │   │   ├── imagesVal -> $nnActive_raw/nnUNet_raw/Dataset004_Hippocampus/imagesVal
│   │   │   ├── labelsTr # contains data with ignore label where only patches are annotated
│   │   │   ├── labelsVal -> $nnActive_raw/nnUNet_raw/Dataset004_Hippocampus/labelsVal
│   │   │   └── labelsVal -> $nnActive_raw/nnUNet_raw/Dataset004_Hippocampus/labelsVal
│   ...
...

$nnActive_results
│   ├── nnActive_results
│   │   ├── Dataset000_Hippocampus__patch-20__qs20__unc-random__seed-12345
│   │   │   └── config.json
│   │   │   └── loop_000 # these will be created for validation and performance etc.
```

After the experiment has been set up, it can now be run.
```bash
nnactive run_experiment --experiment Hippocampus__patch-20_20_20__qs-40__unc-random__seed-12345
```



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
    // id to annotated dataset 
}
```


## Additional labels path
Additional labels can be added in the `addTr` folder, these will then be written to the `labelsTr` folder.

Implementation checking overlap will be implemented in the future version.

## Active Learning Integration

The annotated data for each loop is saved in the `loop_XXX.json` file situated in the respective nnUNet_raw folder for each experiment.
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


## Contributing

- *Run `pre-commit install` every time you clone the repo*
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