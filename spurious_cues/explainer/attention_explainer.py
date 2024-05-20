from spurious_cues.explainer.base_explainer import BaseExplainer
from typing import Dict
import torch


class AttentionExplainer(BaseExplainer):
    def __init__(self, model_checkpoint: str, dataset: str, top_k=5):
        super().__init__(model_checkpoint, dataset, top_k)

    def explain_sentence(self, sent) -> Dict[str, float]:
        tokens = self.tokenizer(
            sent,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        # model inference
        with torch.no_grad():
            output = self.model(**tokens, output_attentions=True)

        no_tokens = output.attentions[0].shape[-1]
        top_k = self.top_k if self.top_k <= no_tokens else no_tokens
        attentions = torch.mean(output[-1][-1], dim=1, keepdim=False)[
            :, 0, :
        ].topk(
            top_k,
        )  # get the top k tokens with the highest attention scores
        # get the decoded tokens and attention scores for the top k tokens in the sentence
        # (excluding special tokens)
        output = {
            self.tokenizer.decode(
                tokens["input_ids"][:, seq_pos_idx].item()
            ): att_value
            for seq_pos_idx, att_value in zip(
                attentions.indices[-1].tolist(),
                attentions.values[-1].tolist(),
            )
            if tokens["input_ids"][:, seq_pos_idx].item() > 100
        }
        return output
