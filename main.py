import torch
import clip
from PIL import Image

#加载模型
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"正在使用设备: {device} 加载模型...")
model, preprocess = clip.load("ViT-B/32", device=device)

image_filename = "test_image.jpg"  
descriptions = ["Bulbasaur", "Pikachu", "Squirtle"]

#处理图片和文本
image = preprocess(Image.open(image_filename)).unsqueeze(0).to(device)
text = clip.tokenize(descriptions).to(device)

#计算匹配度
with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("\n----------------- 测试结果 -----------------")
for i, desc in enumerate(descriptions):
    print(f"文本: '{desc}' \t匹配概率: {probs[0][i]:.4f}")

best_match_idx = probs.argmax()
print(f"\n模型认为这张图是: '{descriptions[best_match_idx]}'")