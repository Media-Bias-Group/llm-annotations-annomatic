from abc import ABC, abstractmethod


class BaseTest(ABC):

    def __init__(self):
        pass

    def prepare_test_data(self):
        pass

    def test(self,model):
        # if data not available, prepare test data
        pass