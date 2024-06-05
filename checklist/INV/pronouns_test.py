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

        d = load_dataset("mediabiasgroup/BABE")["test"]
        texts = d["text"]
        print("Replacing Named Entities...")
        ner_free_texts = list(map(self.replace_named_entities, texts))

        self.test_data = pd.DataFrame.from_dict(
            {
                "text_orig": texts,
                "text_ner_free": ner_free_texts,
                "label": d["label"],
            }
        )

    def test(self, test_data):
        # first only evaluate on original test data, and keep only the instances
        # where model is correct in the first place
        test_data["preds"] = self.make_predictions(target_col="text_orig",data=test_data)
        test_data = test_data[test_data.preds == test_data.label]
        test_data["preds"] = self.make_predictions(target_col="text_ner_free",data=test_data)
        return test_data
