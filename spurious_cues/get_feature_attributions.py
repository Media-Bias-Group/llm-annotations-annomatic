from spurious_cues.explainer.lime_explainer import LimeExplainer

# load model and dataset
babe_base = "anonymous/babe-base-annomatic"
magpie = "anonymous/magpie-annomatic"
synth = "anonymous/roberta-anno-lexical-ft"

# dataset = "mediabiasgroup/BABE"
dataset = "horychtom/BASIL" # public version of BASIL dataset


# explain feature attributions
ann = LimeExplainer(babe_base, dataset,split='train')
ann.explain_dataset(class_=1)

# ann = LimeExplainer(magpie, dataset,split='test')
# ann.explain_dataset(class_=1)

ann = LimeExplainer(synth, dataset,split='train')
ann.explain_dataset(class_=1)
