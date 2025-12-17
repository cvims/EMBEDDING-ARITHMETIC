import torch
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer

class PrismVLMAdapter(nn.Module):
    """
    The Prism VLM Adapter module.
    
    This small network takes a deterministic embedding from a frozen CLIP model
    and maps it to the parameters of a Gaussian distribution (mean and log-variance).
    These are the only parameters that will be trained.
    """
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.mean_head = nn.Linear(embedding_dim, embedding_dim)
        self.log_var_head = nn.Linear(embedding_dim, embedding_dim)
    def forward(self, clip_embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu = self.mean_head(clip_embedding)
        log_var = self.log_var_head(clip_embedding)
        return mu, log_var
    
class DebiasAdapter(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.mean_head = nn.Linear(embedding_dim, embedding_dim)
    def forward(self, clip_embedding: torch.Tensor):
        mu = self.mean_head(clip_embedding)
        return mu

class IdentityAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        # self.embedding_dim = embedding_dim
        self.ident = nn.Identity()
    def forward(self, clip_embedding: torch.Tensor):
        return self.ident(clip_embedding)
        
class PrismVLMTextEncoder(nn.Module):
    def __init__(self, clip_model_name = None, 
                 adapter_type = "prism",
                 text_model = None, 
                 tokenizer = None, 
                 embedding_dim = None, 
                 device: str = "cuda",
                 dtype = torch.bfloat16) -> None:
        
        super().__init__()
        self.adapter_type = "identity"
        self.device = device
        self.adapter_type = adapter_type
        self.dtype = dtype
        # if not clip_model_name:
        self.tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
        self.text_encoder = CLIPTextModel.from_pretrained(clip_model_name).to(device = self.device, dtype = self.dtype)
        print("model loaded")
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        
        try:
            self.embedding_dim = self.text_encoder.config.hidden_size
            if not embedding_dim:
                print(f"Overriding the given embedding dim to the actual hidden size {self.embedding_dim} of the text_encoder for compatibility")
        except Exception as e:
            print(f"cannot automatically parse embedding_dim size, setting from the given argument : {e}")
            if not self.embedding_dim:
                self.embedding_dim = embedding_dim
            else:
                raise ValueError

        if self.adapter_type == "identity":
            self.adapter = IdentityAdapter()
        
    def forward(self, prompts):
        self.text_encoder.eval()
        clip_embeddings = self._get_embedding(prompts)
        if self.adapter_type == "prism":
            mu, log_var = self.adapter(clip_embeddings)
            return mu, log_var
        elif (self.adapter_type == "debiaser") or (self.adapter_type == "identity"):
            mu = self.adapter(clip_embeddings)
            return mu
        
        
    
    def _get_embedding(self, prompt):
        inputs = self.tokenizer(
            prompt,
            padding = True,
            truncation = True,
            return_tensors="pt" 
        ).to(self.device)
        with torch.no_grad():
            clip_outputs = self.text_encoder(**inputs)
            clip_embeddings = clip_outputs.pooler_output
        return clip_embeddings
        
    def sample(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)
        return mu + epsilon*std