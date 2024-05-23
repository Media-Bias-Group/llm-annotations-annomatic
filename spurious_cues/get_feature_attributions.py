"""This script uses the Explainer class to explain feature attributions for a given model and dataset."""
from spurious_cues.explainer.lime_explainer import LimeExplainer

# load model and dataset
babe_base = "mediabiasgroup/babe-base-annomatic"
magpie = "mediabiasgroup/magpie-annomatic"
synth = "mediabiasgroup/roberta-anno-lexical-ft"

# dataset = "mediabiasgroup/BABE"
dataset = "horychtom/experiments"


# explain feature attributions
ann = LimeExplainer(babe_base, dataset,split='train')
ann.explain_dataset(class_=1)

# ann = LimeExplainer(magpie, dataset,split='test')
# ann.explain_dataset(class_=1)

ann = LimeExplainer(synth, dataset,split='train')
ann.explain_dataset(class_=1)

# visualize feature attributions
# ann.get_wordcloud(silhouette_path="data/figures/silh1.jpg")
