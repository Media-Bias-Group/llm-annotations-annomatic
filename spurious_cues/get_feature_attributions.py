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
