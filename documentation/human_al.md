# Human as patch selector


Creating loop_json from `{DATAPATH}/patches_manual_selected_{loop}` folder with patch_size=X,Y,Z:
```bash
nnactive human_al_selection_to_loop --raw_folder {DATAPATH} --loop {loop} --patch_size "[{patch_size}]"
```

Create highlighted regions for annotation in `{DATAPATH}/masksTr_boundary_{loop}`
```bash
nnactive manual_query --raw_folder {DATAPATH} --loop {loop} --identify_patches True
```

Create a folder with cropped predictions:
```bash
nnactive manual_crop_pred --raw_folder {DATAPATH} --loop {loop}
```
