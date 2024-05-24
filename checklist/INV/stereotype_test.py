from checklist import BaseTest
import sqlite3
import pandas as pd
from transformers import pipeline
from sklearn.metrics import f1_score,matthews_corrcoef,recall_score


class PrejudiceTest(BaseTest):

    def __init__(self, data_path):
        super().__init__(data_path)


    def initialize_model(self, model: str):
        self.model = pipeline("text-classification", model=model,batch_size=8)


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

        print("Preparing test data...")
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

    
    def compute_metrics(self,y_true,y_preds):
        return matthews_corrcoef(y_true,y_preds)


    def test(self):
        print("Running model on the test...")
        preds =  self.model(self.test_data['text'].tolist())
        self.test_data['preds'] = [int(pred['label'].split('_')[1]) for pred in preds]
        self.test_data.to_csv("checklist/INV/output.csv",index=False)
        print(self.compute_metrics(self.test_data['label'],self.test_data['preds']))




pt = PrejudiceTest("checklist/data")
pt.execute('mediabiasgroup/magpie-annomatic')
