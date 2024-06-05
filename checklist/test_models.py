from checklist.DIR import LoadedTest
from checklist.INV import LocationsTest, PrejudiceTest, PronounsTest
from checklist.MFT import FactualTest

model_list = [
    "mediabiasgroup/babe-base-annomatic",
    "mediabiasgroup/magpie-annomatic",
    "mediabiasgroup/roberta-anno-lexical-ft",
]

test_list = [
    LoadedTest,
    LocationsTest,
    PrejudiceTest,
    PronounsTest,
    FactualTest,
]

for test in test_list:
    for model in model_list:
        print(f"Running {test.__name__} on all models...")
        print(f"running on {model} ...")
        t = test("checklist/data")
        t.execute(model)
