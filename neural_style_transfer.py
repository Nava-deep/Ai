import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import os
from typing import Tuple
import sys
import urllib.request

# Configure logging for debugging and tracking
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VGG19(nn.Module):
    def __init__(self):
        # Initialize VGG19 model for feature extraction
        super(VGG19, self).__init__()
        vgg = models.vgg19(weights='IMAGENET1K_V1').features
        self.layers = nn.ModuleList()
        for layer in vgg:
            if isinstance(layer, nn.Conv2d):
                self.layers.append(layer)
            elif isinstance(layer, nn.ReLU):
                self.layers.append(nn.ReLU(inplace=False))
            elif isinstance(layer, nn.MaxPool2d):
                self.layers.append(layer)

    def forward(self, x):
        # Extract features from VGG19 layers
        features = []
        for layer in self.layers:
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                features.append(x)
        return features

class StyleTransfer:
    def __init__(self, content_path: str, style_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        # Initialize style transfer with content and style images
        self.device = device
        try:
            self.model = VGG19().to(device).eval()
            logger.info("VGG19 model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load VGG19: {str(e)}")
            sys.exit(1)
        self.content_img = self.load_image(content_path).to(device)
        self.style_img = self.load_image(style_path).to(device)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.denorm = transforms.Normalize(mean=[-2.12, -2.04, -1.80], std=[4.37, 4.46, 4.44])

    def load_image(self, path: str, size: int = 512) -> torch.Tensor:
        # Load and preprocess an image for style transfer
        try:
            if not os.path.exists(path):
                logger.error(f"Image not found: {path}")
                raise FileNotFoundError(f"Image not found: {path}")
            img = Image.open(path).convert('RGB')
            w, h = img.size
            if w > h:
                new_w = size
                new_h = int(size * h / w)
            else:
                new_h = size
                new_w = int(size * w / h)
            img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            logger.info(f"Loaded and resized image: {path}")
            return self.transform(img).unsqueeze(0)
        except Exception as e:
            logger.error(f"Failed to load image {path}: {str(e)}")
            raise

    def gram_matrix(self, tensor: torch.Tensor) -> torch.Tensor:
        # Compute Gram matrix for style features
        b, c, h, w = tensor.size()
        features = tensor.view(b * c, h * w)
        gram = torch.mm(features, features.t())
        return gram.div(b * c * h * w)

    def get_features(self, img: torch.Tensor) -> list:
        # Extract features from the input image using VGG19
        return self.model(img)

    def style_transfer(self, iterations: int = 300, content_weight: float = 1e5, style_weight: float = 1e10) -> torch.Tensor:
        # Perform neural style transfer
        try:
            target = self.content_img.clone().requires_grad_(True).to(self.device)
            optimizer = optim.LBFGS([target])
            content_features = self.get_features(self.content_img)
            style_features = self.get_features(self.style_img)
            style_grams = [self.gram_matrix(feat) for feat in style_features]

            for i in range(iterations):
                def closure():
                    optimizer.zero_grad()
                    target_features = self.get_features(target)
                    content_loss = torch.mean((target_features[2] - content_features[2]) ** 2)
                    style_loss = 0
                    for j, (t, s) in enumerate(zip(target_features, style_grams)):
                        t_gram = self.gram_matrix(t)
                        style_loss += torch.mean((t_gram - s) ** 2)
                    style_loss /= len(style_grams)
                    total_loss = content_weight * content_loss + style_weight * style_loss
                    total_loss.backward()
                    if i % 50 == 0:
                        logger.info(f"Iteration {i}, Total Loss: {total_loss.item()}")
                    return total_loss

                optimizer.step(closure)

            logger.info("Style transfer completed")
            return target.detach()
        except Exception as e:
            logger.error(f"Style transfer failed: {str(e)}")
            raise

    def save_image(self, tensor: torch.Tensor, output_path: str):
        # Save the styled image to disk
        try:
            img = tensor.cpu().clone().squeeze(0)
            img = self.denorm(img).clamp_(0, 1)
            img = transforms.ToPILImage()(img)
            img.save(output_path)
            logger.info(f"Saved styled image to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save image: {str(e)}")
            raise

def download_image(url: str, filename: str):
    # Download an image from a URL
    try:
        urllib.request.urlretrieve(url, filename)
        logger.info(f"Downloaded image to {filename}")
    except Exception as e:
        logger.error(f"Failed to download image: {str(e)}")
        raise

def main():
    # Execute style transfer with sample images
    content_url = "https://images.unsplash.com/photo-1507525428034-b723cf961d3e"
    style_url = "https://upload.wikimedia.org/wikipedia/commons/0/0a/The_Starry_Night_-_Vincent_van_Gogh.jpg"
    content_path = "content.jpg"
    style_path = "style.jpg"
    output_path = "output_styled_image.jpg"

    download_image(content_url, content_path)
    download_image(style_url, style_path)

    st = StyleTransfer(content_path, style_path)
    styled_image = st.style_transfer(iterations=500)
    st.save_image(styled_image, output_path)

    print(f"\n{'='*50}\nStyle Transfer Complete\n{'='*50}")
    print(f"Content Image: {content_path}")
    print(f"Style Image: {style_path}")
    print(f"Output Image: {output_path}")

if __name__ == "__main__":
    main()
