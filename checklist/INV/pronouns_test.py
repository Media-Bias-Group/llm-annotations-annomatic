from checklist import BaseTest
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
)
from datasets import load_dataset
import spacy


class PronounsTest(BaseTest):

    def __init__(self, data_path):
        super().__init__(data_path)
        self.pos_pipe = spacy.load("en_core_web_sm")

    def compute_metrics(self, y_true, y_preds):
        return accuracy_score(y_true, y_preds)

    def replace_named_entities(self, text):
        doc = self.pos_pipe(text)
        replaced_text = text

        # Create a mapping of named entities to pronouns
        entity_pronoun_map = {
            "PERSON": "they",
            "ORG": "it",
        }

        for ent in doc.ents:
            if ent.label_ in entity_pronoun_map:
                entity = ent.text
                pronoun = entity_pronoun_map[ent.label_]
                pronoun = (
                    pronoun[0].upper() + pronoun[1:]
                    if replaced_text.startswith(entity)
                    else pronoun
                )
                replaced_text = replaced_text.replace(entity, pronoun)

        return replaced_text

    def prepare_test_data(self):
        """
        Prepares the test data by retrieving tables from the 'minorities.db' database
        and storing them in a dictionary.
        """

        print("Preparing test data...")
        d = load_dataset("mediabiasgroup/BABE")["test"]
        texts = d["text"]
        print("Replacing Named Entities...")
        ner_free_texts = list(map(self.replace_named_entities, texts))

        self.test_data = pd.DataFrame.from_dict(
            {"text_orig": texts,"text_ner_free":ner_free_texts,"label": d["label"]}
        )

    def test(self):
        print("Running model on the test...")
        self.test_data["preds_orig"] = self.make_predictions(target_col="text_orig")
        self.test_data["preds_ner_free"] = self.make_predictions(target_col="text_ner_free")
        # only keep correct predictions in the first place
        self.test_data = self.test_data[self.test_data.preds_orig == self.test_data.label]
        self.test_data.to_csv("checklist/INV/pronouns_out.csv", index=False)
        print(
            self.compute_metrics(
                self.test_data["label"], self.test_data["preds_ner_free"]
            )
        )


pt = PronounsTest("checklist/data")
pt.execute("mediabiasgroup/babe-base-annomatic")
