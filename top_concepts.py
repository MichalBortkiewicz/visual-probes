# %%

import numpy as np
import pandas as pd
import os

from config import ROOT_DIR
from utils.script_generator import CLASSES

# test_classes = ["zebra", "dingo", "bison", "koala", "jaguar", "chimpanzee", "hog", "hamster", "lion", "beaver", "lynx", "convertible", "sports_car", "airliner", "jeep", "passenger_car", "steam_locomotive", "cab", "garbage_truck", "warplane", "ambulance", "police_van", "planetarium", "castle", "church", "mosque", "triumphal_arch", "barn", "stupa", "boathouse", "suspension_bridge", "steel_arch_bridge", "viaduct", "sax", "flute", "cornet", "panpipe", "drum", "cello", "acoustic_guitar", "grand_piano", "banjo", "maraca", "chime", "Granny_Smith", "fig", "custard_apple", "banana", "corn", "lemon", "pomegranate", "pineapple", "jackfruit", "strawberry", "orange"]
test_classes = CLASSES
assert len(test_classes) == 50


# %%


def concept_filtering(path_to_concepts, list_of_classes: list):
    all_files_with_sumarries = list(
        map(
            lambda x: os.path.join(
                path_to_concepts,
                f"results_{x}",
                "results_summaries",
                "ace_results.txt",
            ),
            list_of_classes,
        )
    )
    all_tcav_scores_list_of_dataframes = [
        pd.read_table(
            x, sep=",|:", engine="python", names=["model", "concept", "tcav", "pvalue"]
        ).dropna()[["concept", "tcav"]]
        for x in all_files_with_sumarries
    ]

    best_per_class = pd.concat(
        [
            x.sort_values(by=["tcav"], ascending=False).iloc[:1]
            for x in all_tcav_scores_list_of_dataframes
        ]
    )

    all_tcav_scores = pd.concat(all_tcav_scores_list_of_dataframes).sort_values(
        by=["tcav"], ascending=False
    )

    return pd.concat([best_per_class, all_tcav_scores]).drop_duplicates().iloc[:100]


top_concepts = concept_filtering(
    "/home/mbortkie/cl_probing/continual-probing/results", test_classes
)


# %%
top_concepts.to_csv(os.path.join(ROOT_DIR, "visual-probes", "top_concepts_df.csv"), index=False)