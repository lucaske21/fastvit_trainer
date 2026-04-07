import argparse
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from models.builder import create_fastvit_model
from utils.utils import load_config
import json

def main():
    parser = argparse.ArgumentParser(description='FastViT Inference')
    parser.add_argument('--config', default='configs/base_config.yaml', type=str, help='Path to config file')
    parser.add_argument('--checkpoint', required=True, type=str, help='Path to model checkpoint')
    parser.add_argument('--image', required=True, type=str, help='Path to image file')
    parser.add_argument('--classes', type=str, help='Path to classes JSON file')
    args = parser.parse_args()

    # 加載配置
    config = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 創建模型並加載權重
    model = create_fastvit_model(config, config['num_classes']).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()

    # 預處理圖像
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.get('mean', [0.485, 0.456, 0.406]), 
                             std=config.get('std', [0.229, 0.224, 0.225]))
    ])

    img = Image.open(args.image).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)

    # 推理
    with torch.no_grad():
        output = model(img_tensor)
        probs = F.softmax(output, dim=1)
        top5_prob, top5_idx = torch.topk(probs, 5)

    # 加載類別名稱
    class_names = None
    if args.classes:
        with open(args.classes, 'r') as f:
            class_names = json.load(f)

    # 輸出結果
    print(f"Top 5 Predictions for {args.image}:")
    for i in range(5):
        idx = top5_idx[0][i].item()
        prob = top5_prob[0][i].item()
        name = class_names[str(idx)] if class_names else f"Class {idx}"
        print(f"{i+1}: {name} ({prob*100:.2f}%)")

if __name__ == '__main__':
    main()
