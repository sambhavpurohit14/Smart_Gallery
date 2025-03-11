import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from transformers import BertModel, BertTokenizer
from torchvision.models import vit_b_16, ViT_B_16_Weights
from PIL import Image
import numpy as np
import io

class ImageEncoder(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        self.model = vit_b_16(weights=weights)
        
        for param in self.model.parameters():
            param.requires_grad = False
        
        for param in self.model.encoder.layers[-4:].parameters():
            param.requires_grad = True
        
        self.model.heads.head = nn.Sequential(
            nn.Linear(self.model.heads.head.in_features, out_dim),
            nn.LayerNorm(out_dim)
        )
    
    def forward(self, x):
        return self.model(x)

class TextEncoder(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.model = BertModel.from_pretrained('bert-base-uncased')
        
        for param in self.model.parameters():
            param.requires_grad = False
        
        for layer in self.model.encoder.layer[-2:]:
            for param in layer.parameters():
                param.requires_grad = True
        
        self.projection = nn.Sequential(
            nn.Linear(768, out_dim),
            nn.LayerNorm(out_dim)
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand_as(hidden_states)
        summed_hidden_states = hidden_states * mask
        token_counts = mask.sum(dim=1, keepdim=True)
        avg_pooling = summed_hidden_states.sum(dim=1) / token_counts.clamp(min=1e-9)
        return self.projection(avg_pooling)

class CLIPModel(nn.Module):
    def __init__(self, embedding_dim=512):
        super().__init__()
        self.image_encoder = ImageEncoder(embedding_dim)
        self.text_encoder = TextEncoder(embedding_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    
    def forward(self, images, caption_ids, caption_mask):
        image_features = self.image_encoder(images)
        text_features = self.text_encoder(caption_ids, caption_mask)
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        return torch.sum(image_features * text_features, dim=-1) * self.logit_scale.exp().clamp(max=100)

class CLIPFeatureExtractor:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel().to(self.device)
        
        with open(model_path, 'rb') as f:
            buffer = io.BytesIO(f.read())
        
        state_dict = torch.load(buffer, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    def extract_image_features(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            image_features = self.model.image_encoder(image)
            image_features = F.normalize(image_features, dim=-1)
        
        return image_features.cpu().numpy()
