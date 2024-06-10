from checklist import BaseTest
import sqlite3
import pandas as pd
from sklearn.metrics import matthews_corrcoef


class PrejudiceTest(BaseTest):

    def __init__(self, data_path):
        super().__init__(data_path)

    def create_category_df(
        self, category: str, df: pd.DataFrame, template: pd.DataFrame
    ):
        """
        Create a new DataFrame by replacing placeholders in a template DataFrame with examples from another DataFrame.

        Args:
            category (str): The category to create the DataFrame for.
            df (pd.DataFrame): The DataFrame containing the examples.
            template (pd.DataFrame): The template DataFrame with placeholders.

        Returns:
            pd.DataFrame: The new DataFrame with replaced placeholders.

        """
        rowlist = []

        template = template[template["category"] == category]

        for _, instance_ in template.iterrows():

            template_text = instance_["text"]
            label = instance_["label"]

            for _, row in df.iterrows():
                minority = row["minority"]
                example = row["example"]

                text = template_text.replace("[" + category + "]", example)
                rowlist.append(
                    {
                        "category": category,
                        "category_minority": minority,
                        "text": text,
                        "label": label,
                    }
                )

        return pd.DataFrame(rowlist)

    def prepare_test_data(self):
        """
        Prepares the test data by retrieving tables from the 'minorities.db' database
        and storing them in a dictionary.
        """

        db = sqlite3.connect(f"{self.data_path}/minorities.db")
        query = "SELECT name FROM sqlite_master WHERE type='table';"
        tables = pd.read_sql_query(query, db)["name"].tolist()

        dataframes = {}

        for table in tables:
            dataframes[table] = pd.read_sql_query(
                f"SELECT * FROM '{table}'", db
            )

        template = dataframes.pop("template_sentences")

        category_dfs = []
        for category, df in dataframes.items():
            category_df = self.create_category_df(category, df, template)
            category_dfs.append(category_df)

        self.test_data = pd.concat(category_dfs)


    def test(self,test_data):
        test_data["preds"] = self.make_predictions(data=test_data)
        return test_data