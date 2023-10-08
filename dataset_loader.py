from torch.utils.data import Dataset
from PIL import Image, ImageOps
import os


class MemeDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        """
        :param root_dir: Directory with all the images.
        :param transform: Optional transform to be applied on a sample.
        """

        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name).convert('RGB')

        # Resize and padding
        max_size = 128
        w_percent = max_size / max(image.size)
        new_size = tuple([int(dim * w_percent) for dim in image.size])
        image = image.resize(new_size, 1)  # Using BICUBIC for resampling

        # Padding
        delta_w = max_size - new_size[0]
        delta_h = max_size - new_size[1]
        padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
        image = ImageOps.expand(image, padding)

        if self.transform:
            image = self.transform(image)

        return image
