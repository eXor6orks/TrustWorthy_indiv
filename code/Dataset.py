import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# Transformations pour l'entrainement (avec augmentation)
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    # transforms.RandomRotation(15),
    # transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Transformations pour le test (Juste Resize + Normalize)
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class Dataset(Dataset):
    def __init__(self, root_dir, split="Train", transform=None):
        """
        root_dir: path vers le dossier Dataset/
        split: 'Train' ou 'Test'
        transform: torchvision transforms
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        self.classes = {
            "Benign": 0,
            "Malignant": 1,
        }

        self.samples = []
        self._load_samples()

    def _load_samples(self):
        split_dir = os.path.join(self.root_dir, self.split)

        for class_name, label in self.classes.items():
            class_dir = os.path.join(split_dir, class_name)

            if not os.path.isdir(class_dir):
                continue

            for file_name in os.listdir(class_dir):
                if file_name.lower().endswith((".png", ".jpg", ".jpeg")):
                    img_path = os.path.join(class_dir, file_name)
                    self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)

class DatasetClient(Dataset):
    def __init__(self, root_dir, split="Train", transform=None, cid=0, nb_clients=3):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.classes = {
            "Benign": 0,
            "Malignant": 1,
        }
        
        all_samples = []
        self._load_all_samples(all_samples)
        
        # Division des données : chaque client prend une tranche (slice)
        # On mélange avec un seed fixe pour que la partition soit cohérente
        import random
        random.seed(42) 
        random.shuffle(all_samples)
        
        # Calcul de la tranche pour ce client spécifique
        size = len(all_samples) // nb_clients
        start = cid * size
        end = start + size if cid < nb_clients - 1 else len(all_samples)
        
        self.samples = all_samples[start:end]

    def _load_all_samples(self, sample_list):
        split_dir = os.path.join(self.root_dir, self.split)
        for class_name, label in self.classes.items():
            class_dir = os.path.join(split_dir, class_name)
            if os.path.isdir(class_dir):
                for f in os.listdir(class_dir):
                    if f.lower().endswith((".png", ".jpg", ".jpeg")):
                        sample_list.append((os.path.join(class_dir, f), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)