import argparse
import os
import csv
import torch
import torchvision.transforms as transforms
import yaml
import numpy as np
from tqdm import tqdm
import time
import random
from sklearn.metrics import average_precision_score, accuracy_score
from PIL import Image
from torch.utils.data import Dataset, DataLoader

# Import DRCT-specific modules
from network.models import get_models
from data.transform import create_val_transforms
from data.dataset import GenImage_LIST, CLASS2LABEL_MAPPING

def set_seed(seed=42):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_parser():
    parser = argparse.ArgumentParser(description="DRCT Testing on Benchmark Datasets")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--model_name", default='convnext_base_in22k', help="Model architecture name", type=str)
    parser.add_argument("--model_path", default=None, help="Path to pretrained model weights", type=str)
    parser.add_argument("--embedding_size", default=1024, help="Embedding size for DRCT model", type=int)
    parser.add_argument("--max_sample", type=int, default=500, help="Max samples per dataset")
    parser.add_argument("--input_size", default=224, help="Image input size", type=int)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device_id", default='0', help="GPU ID", type=str)
    parser.add_argument("--is_crop", action='store_true', help="Whether to crop images")
    parser.add_argument("--save_bad_case", action="store_true", help="Save misclassified images")
    parser.add_argument("--result_dir", type=str, default="./results", help="Directory to save results")
    parser.add_argument("--num_workers", default=4, help="Number of worker threads", type=int)
    return parser.parse_args()

class ImageDataset(Dataset):
    """Dataset for loading real/fake images"""
    def __init__(self, data_path, label, max_sample, transform, is_geneval=False):
        self.transform = transform
        self.label = label
        self.data_list = self.load_images(data_path, max_sample, is_geneval)
        print(f"Loaded {len(self.data_list)} images from {data_path}")
        
    def load_images(self, data_path, max_sample, is_geneval):
        """Load images from directory"""
        image_exts = ['.jpg', '.jpeg', '.png', '.bmp']
        
        # Handle GenEval datasets which have a specific pattern
        if is_geneval:
            images = []
            for root, _, files in os.walk(data_path):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in image_exts):
                        if 'GPT-ImgEval' not in data_path or not file.startswith('._'):
                            # For GenEval, check if filename contains _0.png (indicating the first image)
                            if 'GPT-ImgEval' in data_path or '_0.png' in file:
                                images.append(os.path.join(root, file))
        else:
            images = []
            for root, _, files in os.walk(data_path):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in image_exts):
                        if not file.startswith('._'):  # Skip hidden files
                            images.append(os.path.join(root, file))
        
        if not images:
            print(f"Warning: No images found in {data_path}")
            return []
            
        # Limit number of samples if needed
        if max_sample > 0 and max_sample < len(images):
            random.shuffle(images)
            images = images[:max_sample]
            
        return images
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        img_path = self.data_list[idx]
        try:
            img = np.array(Image.open(img_path).convert('RGB'))
            img = self.transform(image=img)['image']
            return img, self.label, img_path
        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            # Return a placeholder in case of error
            placeholder = torch.zeros((3, 224, 224))
            return placeholder, self.label, img_path

def collate_fn(batch):
    """Custom collate function to filter out None values"""
    batch = [item for item in batch if item is not None]
    if not batch:
        return torch.empty(0, 3, 224, 224), torch.empty(0, dtype=torch.int64), []
    
    images, labels, paths = zip(*batch)
    images = torch.stack(images)
    labels = torch.tensor(labels)
    return images, labels, paths

def validate(model, loader, gpu_id, save_incorrect=False, save_dir=None):
    """Validate model on dataset"""
    model.eval()
    y_true, y_pred = [], []
    incorrect_paths = []
    
    with torch.no_grad():
        for images, labels, paths in tqdm(loader, desc="Validating"):

            if images.size(0) == 0:
                continue
                
            images = images.cuda(gpu_id)
            outputs = model(images)
            
            # # For binary classification
            # if hasattr(outputs, 'sigmoid'):
            #     probabilities = outputs.sigmoid().cpu().numpy().flatten()
            # else:
            probabilities = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy().flatten()
                
            predictions = (probabilities > 0.5).astype(int)
            
            # Record incorrect predictions for fake images
            if save_incorrect:
                for i, (label, pred, path) in enumerate(zip(labels, predictions, paths)):
                    if label == 1 and pred == 0:  # Fake classified as real
                        incorrect_paths.append((path, probabilities[i]))
            
            y_true.extend(labels.numpy())
            y_pred.extend(probabilities)
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate metrics
    r_acc = accuracy_score(y_true[y_true==0], (y_pred[y_true==0] > 0.5).astype(int))
    f_acc = accuracy_score(y_true[y_true==1], (y_pred[y_true==1] > 0.5).astype(int))
    acc = accuracy_score(y_true, (y_pred > 0.5).astype(int))
    
    # Save incorrect examples if requested
    if save_incorrect and incorrect_paths and save_dir:
        os.makedirs(save_dir, exist_ok=True)
        csv_file = os.path.join(save_dir, "incorrect_fakes.csv")
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Path', 'Score'])
            for path, score in incorrect_paths[:10]:  # Limit to 10 examples
                writer.writerow([path, score])
                
    return r_acc, f_acc, acc

def main():
    args = get_parser()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_id
    set_seed()
    
    # Create results directory
    os.makedirs(args.result_dir, exist_ok=True)
    
    # Load YAML config
    with open(args.config, 'r') as f:
        dataset_configs = yaml.safe_load(f)
    
    # Load model
    model = get_models(model_name=args.model_name, num_classes=2, embedding_size=args.embedding_size)
    if args.model_path:
        model.load_state_dict(torch.load(args.model_path, map_location='cpu'), strict=True)
        print(f"Loaded model from {args.model_path}")
    else:
        print("No model path provided, using random weights")
    
    model = model.cuda(int(args.device_id.split(',')[0]))
    model.eval()
    
    # Create transforms
    transform = create_val_transforms(size=args.input_size, is_crop=args.is_crop)
    
    # Results dictionary for storing metrics
    results = {}
    results_file = os.path.join(args.result_dir, "results.csv")
    
    # Process each dataset category
    for category, datasets in dataset_configs.items():
        print(f"\n{'='*50}\nProcessing {category} datasets\n{'='*50}")
        
        # Get common real path if defined at category level
        real_path = datasets.get('real')
        
        for dataset_name, dataset_config in datasets.items():
            if dataset_name == 'real':
                continue  # Skip the real path entry
                
            print(f"\n{'-'*40}\nTesting {category}/{dataset_name}\n{'-'*40}")
            
            # For DRCT, use common real path
            if category == 'DRCT':
                dataset_real_path = real_path
                if isinstance(dataset_config, dict) and 'fake' in dataset_config:
                    dataset_fake_path = dataset_config['fake']
                else:
                    continue
            # For GenImage, each dataset has its own real/fake paths
            elif category == 'GenImage' or category == 'Chameleon':
                if isinstance(dataset_config, dict):
                    dataset_real_path = dataset_config.get('real')
                    dataset_fake_path = dataset_config.get('fake')
                else:
                    continue
            # For GenEval, only fake paths are provided
            elif category == 'GenEval':
                dataset_real_path = None
                dataset_fake_path = dataset_config.get('fake')
            else:
                continue
                
            # Calculate metrics for real images if available
            if dataset_real_path and os.path.exists(dataset_real_path):
                real_dataset = ImageDataset(
                    data_path=dataset_real_path,
                    label=0,
                    max_sample=args.max_sample,
                    transform=transform,
                    is_geneval=False
                )
                
                if len(real_dataset) > 0:
                    real_loader = DataLoader(
                        real_dataset, 
                        batch_size=args.batch_size, 
                        shuffle=False, 
                        num_workers=args.num_workers,
                        collate_fn=collate_fn
                    )
                    
                    r_acc, _, _ = validate(
                        model, 
                        real_loader, 
                        gpu_id=int(args.device_id.split(',')[0]), 
                        save_incorrect=False
                    )
                    print(f"Real accuracy: {r_acc:.4f}")
                else:
                    r_acc = None
                    print("No real images found or loaded")
            else:
                r_acc = None
                print("No real path provided or path does not exist")
                
            # Calculate metrics for fake images
            if dataset_fake_path and os.path.exists(dataset_fake_path):
                fake_dataset = ImageDataset(
                    data_path=dataset_fake_path,
                    label=1,
                    max_sample=args.max_sample,
                    transform=transform,
                    is_geneval=(category == 'GenEval')
                )
                
                if len(fake_dataset) > 0:
                    fake_loader = DataLoader(
                        fake_dataset, 
                        batch_size=args.batch_size, 
                        shuffle=False, 
                        num_workers=args.num_workers,
                        collate_fn=collate_fn
                    )
                    
                    save_dir = os.path.join(args.result_dir, f"bad_cases/{category}_{dataset_name}") if args.save_bad_case else None
                    
                    _, f_acc, _ = validate(
                        model, 
                        fake_loader, 
                        gpu_id=int(args.device_id.split(',')[0]), 
                        save_incorrect=args.save_bad_case,
                        save_dir=save_dir
                    )
                    print(f"Fake accuracy: {f_acc:.4f}")
                else:
                    f_acc = None
                    print("No fake images found or loaded")
            else:
                f_acc = None
                print("No fake path provided or path does not exist")
                
            # Calculate average accuracy if both metrics are available
            if r_acc is not None and f_acc is not None:
                avg_acc = (r_acc + f_acc) / 2
                print(f"Average accuracy: {avg_acc:.4f}")
            else:
                avg_acc = None
                
            # Store results
            key = f"{category}-{dataset_name}"
            results[key] = (r_acc, f_acc, avg_acc)
            
    # Save results to CSV
    with open(results_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header row
        writer.writerow(['Dataset', 'Real Accuracy', 'Fake Accuracy', 'Average Accuracy'])
        
        # Write result rows
        for dataset, (r_acc, f_acc, avg_acc) in results.items():
            writer.writerow([
                dataset,
                f"{r_acc:.4f}" if r_acc is not None else "N/A",
                f"{f_acc:.4f}" if f_acc is not None else "N/A",
                f"{avg_acc:.4f}" if avg_acc is not None else "N/A"
            ])
    
    print(f"\nResults saved to {results_file}")
    
    # Also save a transposed version for easier analysis
    transpose_file = os.path.join(args.result_dir, "results_transposed.csv")
    
    with open(transpose_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Headers: metric name + dataset names
        dataset_names = list(results.keys())
        writer.writerow(['Metric'] + dataset_names)
        
        # Real accuracy row
        real_accs = [results[d][0] if results[d][0] is not None else "N/A" for d in dataset_names]
        writer.writerow(['Real Accuracy'] + real_accs)
        
        # Fake accuracy row
        fake_accs = [results[d][1] if results[d][1] is not None else "N/A" for d in dataset_names]
        writer.writerow(['Fake Accuracy'] + fake_accs)
        
        # Average accuracy row
        avg_accs = [results[d][2] if results[d][2] is not None else "N/A" for d in dataset_names]
        writer.writerow(['Average Accuracy'] + avg_accs)
    
    print(f"Transposed results saved to {transpose_file}")

if __name__ == "__main__":
    main()