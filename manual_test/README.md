# Manual Tests for nnActive Development

## Data Test for checking use_mask_for_norm and preprocessing
File:
```
python brats_integrity_mask_for_norm
```
1. Checks that plan files share important config values across original dataset before before resampling, after resampling and with ignore labels. 
2. Checks that preprocessed data is identical for preprocessed before and after resampling.
3. Checks that data in preprocessed npz files is identical before resampling, after resampling and with ignore labels.