from abc import ABC, abstractmethod
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch


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

    def initialize_model(self, model: str):
        """
        Initializes the model for sequence classification.

        Args:
            model (str): The name or path of the pre-trained model.

        """
        self.model = AutoModelForSequenceClassification.from_pretrained(model)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
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

    def execute(self, model: str):
        """
        Executes the test on the given model.

        Args:
            model: The model to be tested.

        Returns:
            The result of the test.
        """
        if self.test_data is None:
            self.prepare_test_data()

        self.initialize_model(model)

        result = self.test()

        return result
