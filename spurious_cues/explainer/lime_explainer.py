from spurious_cues.explainer.base_explainer import BaseExplainer
from lime.lime_text import LimeTextExplainer
from typing import Dict
import torch.nn.functional as F
import torch
from transformers import pipeline
import numpy as np

class LimeExplainer(BaseExplainer):
    def __init__(self, model_checkpoint: str, dataset: str, top_k=5,split='train'):
        super().__init__(model_checkpoint, dataset, top_k,split)
        self.text_explainer = LimeTextExplainer(class_names=self.class_names)
        self.hf_predictor =  pipeline("text-classification", model=model_checkpoint,device=0 if torch.cuda.is_available() else None)

    def explain_sentence(self, sent) -> Dict[str, float]:
        
        def predictor(texts):
            label2id = self.hf_predictor.model.config.label2id
            output = self.hf_predictor(texts)
            probs = []
            for cls in output:
                label = label2id[cls['label']]
                logit = cls['score']

                probabilities = [0,0]
                probabilities[label] = logit
                probabilities[1-label] = 1-logit
                probs.append(probabilities)
            return np.array(probs)

        exp = self.text_explainer.explain_instance(
            sent,
            predictor,
            num_features=5,
            num_samples=100,
        )
        return {k: v for k, v in exp.as_list()}
