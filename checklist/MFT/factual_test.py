from checklist import BaseTest
import sqlite3
import pandas as pd
from transformers import pipeline
from sklearn.metrics import (
    f1_score,
    matthews_corrcoef,
    recall_score,
    accuracy_score,
)
from datasets import load_dataset, Dataset


class FactualTest(BaseTest):

    def __init__(self, data_path):
        super().__init__(data_path)

    def compute_metrics(self, y_true, y_preds):
        return accuracy_score(y_true, y_preds)

    def prepare_test_data(self):
        """
        Prepares the test data by retrieving tables from the 'minorities.db' database
        and storing them in a dictionary.
        """

        print("Preparing test data...")
        d1 = load_dataset("notrichardren/easy_qa")["validation"].to_pandas()
        d2 = load_dataset("truthful_qa", "generation")[
            "validation"
        ].to_pandas()

        d1 = (d1["question"] + " " + d1["right_answer"]).tolist()
        d2 = d2["best_answer"].tolist()

        texts = d1 + d2
        self.test_data = pd.DataFrame.from_dict(
            {"text": texts, "label": [0] * len(texts)}
        )

    def test(self):
        print("Running model on the test...")
        self.test_data["preds"] = self.make_predictions()
        self.test_data.to_csv("checklist/MFT/factual_out.csv", index=False)
        print(
            self.compute_metrics(
                self.test_data["label"], self.test_data["preds"]
            )
        )


pt = FactualTest("checklist/data")
pt.execute("mediabiasgroup/babe-base-annomatic")
