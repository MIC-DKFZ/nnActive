# nnActive Playground

Scripts for nnActive development

Install with
```bash
pip install -e '.[dev]'
```

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

## Active Learning Workflow
nnUNet-Loop:
for each fold X in (0, 1, 2, 3, 4):
```bash
nnUNetv2_predict -d 504 -c 3d_fullres -i ${nnUNet_raw}/Dataset504_Hippocampus-partanno/imagesTr -o ${nnUNet_results}/Dataset504_Hippocampus-partanno/predTr/fold_X -tr nnUNetDebugTrainer --save_probabilities -f X
```
Then aggregate to uncertainties
```bash
python calculate_uncertainties.py -p ${nnUNet_results}/Dataset504_Hippocampus-partanno/predTr
```
Uncertainties are now in folder: `${nnUNet_results}/Dataset504_Hippocampus-partanno/predTr/uncertainties` 

These uncertainties are now aggregated:
```bash
cd uncertainty_aggregation
python aggregate_uncertainties.py -i ${nnUNet_results}/Dataset504_Hippocampus-partanno/predTr/uncertainties -d ${nnUNet_raw}/Dataset504_Hippocampus-partanno/dataset.json
```
These uncertainties are now aggregated in the folder: `${nnUNet_results}/Dataset504_Hippocampus-partanno/predTr/uncertainties/aggregated_uncertainties`

These are used to query patches for loop X:
```bash
python query_patches.py -i ${nnUNet_results}/Dataset504_Hippocampus-partanno/predTr/uncertainties/aggregated_uncertainties -u mutual_information -n 20 -o ${nnUNet_raw}/Dataset504_Hippocampus-partanno -l X
```
Then two files are created in `${nnUNet_raw}/Dataset504_Hippocampus-partanno`:
1. `loop_00X.json`
2. `{uncertainty_type}_loop00X.json`

Now the dataset needs to be updated according to all `loop_XXX.json` files:
```bash

```

