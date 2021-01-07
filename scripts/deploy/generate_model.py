# Author: Aman Tiwari
# Usage: python generate_model.py -f fintuned.pt -d dict.ltr.txt -o final_model.pt
# return model.pt with keys(model, target_dict) use these as new assignments

import torch
import argparse
from fairseq import utils
from fairseq.models import BaseFairseqModel
from fairseq.data import Dictionary
from fairseq.models.wav2vec.wav2vec2_asr import Wav2VecEncoder
from fairseq.models.wav2vec import Wav2Vec2CtcConfig
from fairseq.tasks import FairseqTask, setup_task


class Wav2VecCtc(BaseFairseqModel):
    def __init__(self, cfg: Wav2Vec2CtcConfig, w2v_encoder: BaseFairseqModel):
        super().__init__()
        self.cfg = cfg
        self.w2v_encoder = w2v_encoder

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    @classmethod
    def build_model(cls, cfg: Wav2Vec2CtcConfig, task: FairseqTask):
        """Build a new model instance."""
        w2v_encoder = Wav2VecEncoder(cfg, task.target_dictionary)
        return cls(cfg, w2v_encoder)

    def get_normalized_probs(self, net_output, log_probs):
        """Get normalized probabilities (or log probs) from a net's output."""

        logits = net_output["encoder_out"]
        if log_probs:
            return utils.log_softmax(logits.float(), dim=-1)
        else:
            return utils.softmax(logits.float(), dim=-1)

    def forward(self, **kwargs):
        x = self.w2v_encoder(**kwargs)
        return x


def load_model(model_path, target_dict):
    w2v = torch.load(model_path,map_location=torch.device("cpu"))

    args =  w2v["cfg"].model

    task = setup_task(args)

    model = task.build_model(args)

    model.load_state_dict(w2v["model"], strict=True)

    return [model]

def generate_custom_model(finetuned_path,dictionary_path,final_model_path):
    target_dict = Dictionary.load(dictionary_path)
    model = load_model(finetuned_path, target_dict)
    torch.save(model,final_model_path)

    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert default models to combined model')
    parser.add_argument('-f', '--finetuned_model_path', type=str, help="Fine-tuned Model path")
    parser.add_argument('-d', '--dict', type=str, help="Dict path")
    parser.add_argument('-o', '--output_path', type=str, default='final_model.pt', help= "Final model path")
    args_local = parser.parse_args()

    generate_custom_model(finetuned_path=args_local.finetuned_model_path,dictionary_path=args_local.dict,final_model_path=args_local.output_path)
    
    
