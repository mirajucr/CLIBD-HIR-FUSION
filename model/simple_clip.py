import torch.nn as nn
import torch.nn.functional as F


class SimpleCLIP(nn.Module):
    def __init__(self, dna_encoder, dna_tokenizer, device, open_clip_model):
        super().__init__()
        self.dna_encoder = dna_encoder
        self.dna_tokenizer = dna_tokenizer
        self.device = device
        self.open_clip_model = open_clip_model

    def forward(self, image_input, language_input, dna_input):
        image_output = None
        dna_output = None
        language_output = None

        if self.dna_encoder is not None and dna_input is not None:
            dna_tokens = self.dna_tokenizer(dna_input).to(self.device)
            dna_embedding = self.dna_encoder(dna_tokens)[0].mean(dim=1)
            dna_output = F.normalize(dna_embedding, p=2, dim=-1)

        if self.open_clip_model is not None:
            if image_input is not None:
                image_features = self.open_clip_model.encode_image(image_input)
                image_output = F.normalize(image_features, p=2, dim=-1)
            if language_input is not None:
                text_features = self.open_clip_model.encode_text(language_input)
                language_output = F.normalize(text_features, p=2, dim=-1)

        return image_output, dna_output, language_output


def unwrap(model):
    return model.module if hasattr(model, "module") else model
