# Human as patch selector


Creating loop_json from `{DATAPATH}/patches_manual_selected_{loop}` folder:
```bash
nnactive human_al_selection_to_loop --raw_folder {DATAPATH} --loop {loop}
```

Create highlighted regions for annotation in `{DATAPATH}/masksTr_boundary_{loop}`
```bash
nnactive manual_query --raw_folder {DATAPATH} --loop {loop} --identify_patches True
```

