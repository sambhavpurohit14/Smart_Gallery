import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50
from transformers import BertModel, BertTokenizer
from torchvision import transforms
from PIL import Image
import random
import json

class COCOAlignmentDataset(Dataset):
    def __init__(self, img_dir, annotations_file, transform=None):
        self.img_dir = img_dir
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
    def __len__(self):
        return len(self.annotations['annotations'])
    
    def __getitem__(self, idx):
        ann = self.annotations['annotations'][idx]
        img_path = f"{self.img_dir}/{ann['image_id']:012d}.jpg"
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        # Anchor caption
        caption = ann['caption']
        encoding = self.tokenizer(caption, padding='max_length', 
                                truncation=True, max_length=64,
                                return_tensors='pt')
        
        # Get negative caption (random from dataset)
        neg_idx = random.choice([i for i in range(len(self)) if i != idx])
        neg_caption = self.annotations['annotations'][neg_idx]['caption']
        neg_encoding = self.tokenizer(neg_caption, padding='max_length',
                                    truncation=True, max_length=64,
                                    return_tensors='pt')
        
        return {
            'image': image,
            'caption_ids': encoding['input_ids'].squeeze(0),
            'caption_mask': encoding['attention_mask'].squeeze(0),
            'neg_caption_ids': neg_encoding['input_ids'].squeeze(0),
            'neg_caption_mask': neg_encoding['attention_mask'].squeeze(0)
        }

class ImageEncoder(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.model = resnet50(pretrained=True)
        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, out_dim),
            nn.LayerNorm(out_dim)
        )
        
    def forward(self, x):
        return self.model(x)

class TextEncoder(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.projection = nn.Sequential(
            nn.Linear(768, out_dim),
            nn.LayerNorm(out_dim)
        )
        
    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, 
                           attention_mask=attention_mask)
        return self.projection(outputs.pooler_output)

class CLIPModel(nn.Module):
    def __init__(self, embedding_dim=512):
        super().__init__()
        self.image_encoder = ImageEncoder(embedding_dim)
        self.text_encoder = TextEncoder(embedding_dim)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
    def forward(self, images, caption_ids, caption_mask, 
                neg_caption_ids, neg_caption_mask):
        # Get embeddings
        image_features = self.image_encoder(images)
        text_features = self.text_encoder(caption_ids, caption_mask)
        neg_text_features = self.text_encoder(neg_caption_ids, neg_caption_mask)
        
        # Normalize features
        image_features = nn.functional.normalize(image_features, dim=-1)
        text_features = nn.functional.normalize(text_features, dim=-1)
        neg_text_features = nn.functional.normalize(neg_text_features, dim=-1)
        
        # Scaled pairwise cosine similarities
        logit_scale = self.logit_scale.exp()
        pos_logits = torch.sum(image_features * text_features, dim=-1) * logit_scale
        neg_logits = torch.sum(image_features * neg_text_features, dim=-1) * logit_scale
        
        return pos_logits, neg_logits

def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        images = batch['image'].to(device)
        caption_ids = batch['caption_ids'].to(device)
        caption_mask = batch['caption_mask'].to(device)
        neg_caption_ids = batch['neg_caption_ids'].to(device)
        neg_caption_mask = batch['neg_caption_mask'].to(device)
        
        pos_logits, neg_logits = model(images, caption_ids, caption_mask,
                                     neg_caption_ids, neg_caption_mask)
        
        # Contrastive loss
        labels = torch.ones_like(pos_logits)
        loss = -torch.mean(
            torch.log(torch.sigmoid(pos_logits)) + 
            torch.log(1 - torch.sigmoid(neg_logits))
        )
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def main():
    # Hyperparameters
    batch_size = 32
    learning_rate = 1e-4
    num_epochs = 10
    embedding_dim = 512
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Dataset
    dataset = COCOAlignmentDataset(
        img_dir='path/to/coco/images',
        annotations_file='path/to/coco/annotations.json'
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, 
                          shuffle=True, num_workers=4)
    
    # Model
    model = CLIPModel(embedding_dim=embedding_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(num_epochs):
        loss = train_one_epoch(model, dataloader, optimizer, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}")
        
if __name__ == "__main__":
    main()