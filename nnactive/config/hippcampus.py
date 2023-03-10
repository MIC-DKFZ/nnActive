from nnactive.config import ActiveConfig


def get_test_hippocampus_config():
    return ActiveConfig(
        starting_budget="test",
        trainer="nnUNetTrainer_20epochs",
        query_size=10,
        patch_size=(36, 50, 35),
        uncertainty="mutual_information",
        aggregation="patch",
        model="3d_fullres",
    )
