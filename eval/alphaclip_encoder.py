import torch
import torchvision
import alpha_clip
from PIL import Image
import numpy as np
from torchvision import transforms

class AlphaCLIPNetwork:
    def __init__(self):
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model_type="ViT-L/14@336px"
        self.model, self.preprocess = alpha_clip.load(
            self.clip_model_type, 
            alpha_vision_ckpt_pth="./ckpts/clip_l14_336_grit_20m_4xe.pth", 
            device = self.device)
        
        self.mask_transform = transforms.Compose([
            transforms.ToTensor(), 
            transforms.Resize((336, 336)), # change to (336,336) when using ViT-L/14@336px
            transforms.Normalize(0.5, 0.26)
            ])
        
        self.negatives = ["object", "things", "stuff", "texture"] # constants
        self.positives = []
        with torch.no_grad():
            # tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.positives]).to(device)
            # self.pos_embeds = model.encode_text(tok_phrases)
            tok_phrases = alpha_clip.tokenize(self.negatives).to(self.device)
            self.neg_embeds = self.model.encode_text(tok_phrases) # n_neg, 768
        # self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)
        self.neg_embeds /= self.neg_embeds.norm(dim=-1, keepdim=True)
    
    @torch.no_grad()
    def get_relevancy(self, embed: torch.Tensor, positive_id: int) -> torch.Tensor:
        # embed: h*w, 768
        phrases_embeds = torch.cat([self.pos_embeds, self.neg_embeds], dim=0) # n_pos+n_neg, 768
        p = phrases_embeds.to(embed.dtype)
        output = torch.mm(embed, p.T) # h*w, n_pos+n_neg
        positive_vals = output[..., positive_id : positive_id + 1]
        negative_vals = output[..., len(self.positives) :]
        repeated_pos = positive_vals.repeat(1, len(self.negatives))

        sims = torch.stack((repeated_pos, negative_vals), dim=-1) # h*w, n_neg, 2
        softmax = torch.softmax(10 * sims, dim=-1) # h*w, n_neg, 2
        best_id = softmax[..., 0].argmin(dim=1) # h*w
        return torch.gather(softmax, 1, best_id[..., None, None].expand(best_id.shape[0], len(self.negatives), 2))[
            :, 0, :
        ]
        
    def encode_text(self, text_list):
        text = alpha_clip.tokenize(text_list).to(self.device)
        return self.model.encode_text(text)

    def set_positives(self, text_list):
        self.positives = text_list
        with torch.no_grad():
            tok_phrases = alpha_clip.tokenize(self.positives).to(self.neg_embeds.device)
            self.pos_embeds = self.model.encode_text(tok_phrases)
        self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)
        
    def get_max_across(self, sem_map):
        n_phrases = len(self.positives)
        n_phrases_sims = [None for _ in range(n_phrases)]
        
        n_levels, h, w, _ = sem_map.shape
        clip_output = sem_map.permute(1, 2, 0, 3).flatten(0, 1) # h*w, n_levels, 512

        n_levels_sims = [None for _ in range(n_levels)]
        for i in range(n_levels):
            for j in range(n_phrases):
                probs = self.get_relevancy(clip_output[..., i, :], j)
                pos_prob = probs[..., 0:1] # h*w, 1
                n_phrases_sims[j] = pos_prob 
            n_levels_sims[i] = torch.stack(n_phrases_sims) # n_phrases, h*w, 1
        
        relev_map = torch.stack(n_levels_sims).view(n_levels, n_phrases, h, w)
        return relev_map