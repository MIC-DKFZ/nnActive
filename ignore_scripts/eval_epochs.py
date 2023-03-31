import json
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i", "--input_path")
    args = parser.parse_args()
    path = Path(args.input_path)

    data = []
    for folder in path.iterdir():
        if folder.is_dir():
            epoch = folder.name.split("_")[1]
            print(epoch)
            if len(epoch) > 0:
                epoch = int(epoch.replace("epochs", ""))
            else:
                epoch = 1000
            for fold_dir in folder.iterdir():
                if "fold_" in fold_dir.name:
                    fold = int(fold_dir.name.replace("fold_", ""))
                    summary_dir = fold_dir / "validation" / "summary.json"

                    with open(summary_dir, "r") as file:
                        summary = json.load(file)
                    foreground_dice = summary["foreground_mean"]["Dice"]

                    row = {
                        "epochs": epoch,
                        "folder": folder,
                        "fold": fold,
                        "foreground_dice": foreground_dice,
                    }
                    data.append(row)

    df = pd.DataFrame(data)
    sns.lineplot(data=df, x="epochs", y="foreground_dice", errorbar="sd")
    # plt.savefig("EpochsVsPerformance.pdf")
    plt.savefig("EpochsVsPerformance.png")
    print(df.groupby("epochs")["foreground_dice"].agg(["mean", "std"]))
