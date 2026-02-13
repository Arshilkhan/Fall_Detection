from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image

class FallDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.transform = transform
        self.samples = []

        dropped_empty = 0
        dropped_invalid = 0
        dropped_missing_pair = 0

        for img_path in sorted(self.images_dir.iterdir()):
            if img_path.suffix.lower() not in [".jpg", ".png", ".jpeg"]:
                continue

            label_path = self.labels_dir / f"{img_path.stem}.txt"

            # ❌ Missing label file → drop image
            if not label_path.exists():
                dropped_missing_pair += 1
                continue

            content = label_path.read_text().strip()

            # ❌ Empty label file → drop image
            if content == "":
                dropped_empty += 1
                continue

            parts = content.split()

            # ❌ Invalid label format (must have class + bbox)
            if len(parts) < 5:
                dropped_invalid += 1
                continue

            try:
                class_id = int(parts[0])
            except ValueError:
                dropped_invalid += 1
                continue

            # ❌ Accept only binary labels
            if class_id not in (0, 1):
                dropped_invalid += 1
                continue

            # ✅ Valid (image, label) pair
            self.samples.append((img_path, class_id))

        print(f"[INFO] Loaded {len(self.samples)} valid samples from {images_dir}")
        print(f"       Dropped missing label pairs : {dropped_missing_pair}")
        print(f"       Dropped empty labels        : {dropped_empty}")
        print(f"       Dropped invalid labels      : {dropped_invalid}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            # Skip corrupt images safely
            return self.__getitem__((idx + 1) % len(self.samples))

        if self.transform:
            image = self.transform(image)

        return image, label
