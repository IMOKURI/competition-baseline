import albumentations as A
import cv2
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset


def make_dataset(c, df, transform=None, label=True):
    if False:  # c.params.dataset_type == "xxx":
        pass
    else:
        ds = BaseDataset(c, df, get_transforms(c, transform), label)

    return ds


def make_dataloader(c, ds, shuffle, drop_last):
    dataloader = DataLoader(
        ds,
        batch_size=c.params.batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True,
        drop_last=drop_last,
    )
    return dataloader


class BaseDataset(Dataset):
    def __init__(self, c, df, transform=None, label=True):
        self.df = df
        self.file_names = df["image_id"].values
        self.transform = transform

        self.use_label = label
        if self.use_label:
            self.path = c.settings.dirs.train_image
            self.labels = df["label"].values
        else:
            self.path = c.settings.dirs.test_image

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        file_path = f"{self.path}/{file_name}"
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        if self.use_label:
            label = torch.tensor(self.labels[idx])
            return image, label
        return image


def get_transforms(c, data):
    if data == "train":
        return A.Compose(
            [
                # A.Resize(c.params.size, c.params.size),
                A.RandomResizedCrop(c.params.size, c.params.size),
                A.Transpose(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.2,
                    scale_limit=0.2,
                    rotate_limit=10,
                    border_mode=0,
                    p=0.5,
                ),
                # A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
                # A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
                A.CoarseDropout(p=0.5),
                A.Cutout(
                    max_h_size=int(c.params.size * 0.4),
                    max_w_size=int(c.params.size * 0.4),
                    num_holes=1,
                    p=0.5,
                ),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )

    return A.Compose(
        [
            A.Resize(c.params.size, c.params.size),
            # A.CenterCrop(c.params.size, c.params.size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ]
    )
