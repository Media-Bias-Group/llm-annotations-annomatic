from abc import ABC, abstractmethod
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)
from datasets import Dataset
from torch.utils.data import DataLoader
import torch
from sklearn.metrics import matthews_corrcoef
from tqdm import tqdm
from checklist.utils import get_model_name


class BaseTest(ABC):
    """
    Base class for test cases.

    This class defines the common interface for test cases and provides
    default implementations for some of the methods.

    Attributes:
    - data_path: The path to the test data.
    - test_data: The test data for the current test case.
    """

    def __init__(self, data_path):
        self.data_path = data_path
        self.test_data = None

    def make_predictions(self, data=None, target_col: str = "text"):
        """
        Generates predictions for the test data using the trained model.

        Returns:
            predictions (list): List of predicted labels for the test data.
        """
        if data is None:
            data = self.test_data
        tok = self.tokenizer(
            list(data[target_col]),
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt",
        )
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        testing_dataloader = DataLoader(
            Dataset.from_dict(tok), batch_size=8, collate_fn=data_collator
        )

        predictions = []
        for batch in tqdm(testing_dataloader):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.model(**batch)

            logits = outputs.logits
            predictions.extend(torch.argmax(logits, dim=-1).tolist())

        return predictions

    def compute_metrics(self, y_true, y_preds):
        """
        Compute the Matthews correlation coefficient (MCC) between the true labels and predicted labels.

        Parameters:
        - y_true (array-like): The true labels.
        - y_preds (array-like): The predicted labels.

        Returns:
        - mcc (float): The Matthews correlation coefficient.
        """
        return matthews_corrcoef(y_true, y_preds)

    def initialize_model(self, model_checkpoint: str):
        """
        Initializes the model for sequence classification.

        Args:
            model (str): The name or path of the pre-trained model.

        """
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_checkpoint
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

        self.device = (
            torch.device("cuda:0")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.model.to(self.device)

    @abstractmethod
    def prepare_test_data(self):
        """
        Prepares the test data for the current test case.

        This method should be implemented by subclasses to provide the necessary test data
        for the specific test case.
        """
        raise NotImplementedError

    @abstractmethod
    def test(self):
        """
        This method is used to test the given model.

        Parameters:
        - model: The model to be tested.

        Returns:
        - None
        """
        raise NotImplementedError

    def execute(self, model_checkpoint: str):
        """
        Executes the test on the given model.

        Args:
            model: The model to be tested.

        Returns:
            The result of the test.
        """
        if self.test_data is None:
            print("Preparing test data...")
            self.prepare_test_data()

        self.initialize_model(model_checkpoint=model_checkpoint)

        print("Running model on the test...")
        result = self.test()
        print(f"Saving output to {self.data_path}/out/{self.__class__.__name__.lower()}_{get_model_name(model_checkpoint)}.csv")
        self.test_data.to_csv(
            f"{self.data_path}/out/{self.__class__.__name__.lower()}_{get_model_name(model_checkpoint)}.csv",
            index=False,
        )

        return result
