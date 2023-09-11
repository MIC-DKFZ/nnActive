Rough overview: random-baseline or query-based
Random requires just draws randomly
Random-label requires info about classes


Predictions: Query Inputs
all infos nnUNev2_predict requires
internal state with Top K Areas
query size

necessary:
list of samples
list of already annotated patches


additional for feature based:
save representations
obtain features based on hooks


missing for fold in num_folds
nnUNetv2_predict
Prediction of Files
1. Get list of files
2. for each file do:
  a. preprocess file
3. for each file do: -> predict_logits_from_preprocessed_data
  a. patches= patchify(file)
  b. file_predict=[]
  c. for each patch in patches:
      patch_predict= nnUNet_forward(patch)
      file_predict.append(patch_predict)
  d. file_predict=aggregate(file_predict)
  e. save(file_predict)


what to avoid:
saving whole predictions
keeping multiple predictions in RAM
