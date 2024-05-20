"""This script uses the Explainer class to explain feature attributions for a given model and dataset."""
from spurious_cues.explainer.lime_explainer import LimeExplainer

# load model and dataset
magpie = "mediabiasgroup/magpie-babe-ft"
model_checkpoint = "mediabiasgroup/babe-v3-roberta-fully-trained"
dataset = "mediabiasgroup/BABE-v3"

# explain feature attributions
ann = LimeExplainer(model_checkpoint, dataset)
ann.explain_dataset(class_=1)

# visualize feature attributions
ann.get_wordcloud(silhouette_path="data/figures/silh1.jpg")
