from checklist import BaseTest
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
)
from tqdm import tqdm
from datasets import load_dataset
import spacy
import random


class LocationsTest(BaseTest):

    def __init__(self, data_path, k=5):
        super().__init__(data_path)
        self.pos_pipe = spacy.load("en_core_web_sm")
        self.locations = None
        self.k = k

    def compute_metrics(self, y_true, y_preds):
        return accuracy_score(y_true, y_preds)

    def extract_locs(self, texts):
        """
        Create pool of locations by extracting all locations from a list of texts.

        Args:
            texts (list): A list of texts to extract locations from.
        """
        locs = []
        for text in tqdm(texts):
            text_ = self.pos_pipe(text)
            locs_ = [
                ent.text
                for ent in text_.ents
                if (ent.label_ == "GPE" or ent.label_ == "LOC")
            ]
            locs.extend(locs_)
        return locs

    def sample_new_location_sentences(self, text):
        """
        Replaces location entities in the given text with random locations from a predefined list.

        Args:
            text (str): The input text containing location entities.

        Returns:
            list: A list of texts where location entities have been replaced with random locations.
        """

        text_ = self.pos_pipe(text)
        replaced_text = text
        texts_ = []

        for ent in text_.ents:
            if ent.label_ == "GPE" or ent.label_ == "LOC":
                entity = ent.text
                for _ in range(self.k):
                    replaced_text = replaced_text.replace(
                        entity, random.choice(self.locations)
                    )
                    texts_.append(replaced_text)
        return texts_

    def prepare_test_data(self):
        """
        Prepares the test data by retrieving tables from the 'minorities.db' database
        and storing them in a dictionary.
        """
        print("Preparing test data...")
        d = load_dataset("mediabiasgroup/BABE")["test"].to_pandas()

        print("Extract all locations...")
        texts = d["text"].tolist()
        self.locations = self.extract_locs(texts)

        print("Filtering sentences with locations...")

        texts_orig = []
        texts_locations = []
        labels = []
        for _, row in tqdm(d.iterrows()):
            orig_text = row["text"]
            label = row["label"]
            new_sents = self.sample_new_location_sentences(row["text"])
            if len(new_sents) == 0:
                continue
            texts_orig.extend([orig_text] * len(new_sents))
            texts_locations.extend(new_sents)
            labels.extend([label] * len(new_sents))

        self.test_data = pd.DataFrame.from_dict(
            {
                "text_orig": texts_orig,
                "text_loc": texts_locations,
                "label": labels,
            }
        )

    def test(self):
        print("Running model on the test...")
        orig_data = self.test_data[~self.test_data.text_orig.duplicated()]
        orig_data["preds_orig"] = self.make_predictions(
            target_col="text_orig", data=orig_data
        )
        orig_data = orig_data[orig_data.preds_orig == orig_data.label]
        self.test_data = orig_data[["text_orig"]].merge(
            self.test_data, on="text_orig"
        )

        self.test_data["preds"] = self.make_predictions(target_col="text_loc")
        self.test_data.to_csv("checklist/INV/locations_out.csv", index=False)
        print(
            self.compute_metrics(
                self.test_data["label"], self.test_data["preds"]
            )
        )


pt = LocationsTest("checklist/data")
pt.execute("mediabiasgroup/roberta-anno-lexical-ft")
