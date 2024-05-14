import os
from glob import glob
from pathlib import Path

import yaml
import numpy as np
import torch.utils.data
from PIL import Image

import torch
import lightning as L
from torchvision import transforms

MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)


def worker_init_fn(worker_id):
	np.random.seed(np.random.get_state()[1][0] + worker_id)


class MVTecLightningDatamodule(L.LightningDataModule):
	def __init__(self, data_dir, category, input_size, batch_train, batch_test, shuffle_test=False):
		super().__init__()
		self.data_dir = data_dir
		self.category = category
		self.input_size = input_size
		self.batch_train = batch_train
		self.batch_val = batch_test
		self.shuffle_test = shuffle_test

		self.train_dataset = get_dataset(self.data_dir, self.category, self.input_size, is_train=True)
		self.val_dataset = get_dataset(self.data_dir, self.category, self.input_size, is_train=False)

	def train_dataloader(self):
		return get_dataloader(self.train_dataset, self.batch_train)

	def val_dataloader(self):
		return get_dataloader(self.val_dataset, self.batch_val, shuffle=False)


def get_dataset(data_dir, category, input_size, is_train):
	return MVTecDataset(
		root=data_dir,
		category=category,
		input_size=input_size,
		is_train=is_train,
	)


def get_dataloader(dataset, batch, shuffle=True):
	return torch.utils.data.DataLoader(
		dataset,
		batch_size=batch,
		shuffle=shuffle,
		num_workers=4,
		drop_last=False,
		worker_init_fn=worker_init_fn
	)


class MVTecDataset(torch.utils.data.Dataset):
	def __init__(self, root, category, input_size, is_train=True):
		self.mean = MEAN
		self.std = STD
		self.category = category
		self.un_normalize_transform = transforms.Normalize((-self.mean / self.std).tolist(), (1.0 / self.std).tolist())
		# self.debug_images = []
		self.image_transform = transforms.Compose(
			[
				transforms.Resize(input_size),
				transforms.ToTensor(),
				transforms.Normalize(self.mean.tolist(), self.std.tolist()),
			]
		)
		if is_train:
			self.image_files = glob(
				os.path.join(root, category, "train", "good", "*.png")
			)
		else:
			self.image_files = sorted(glob(os.path.join(root, category, "test", "*", "*.png")))
			self.target_transform = transforms.Compose(
				[
					transforms.Resize(input_size),
					transforms.ToTensor(),
				]
			)

		self.is_train = is_train

	def un_normalize(self, img):
		return self.un_normalize_transform(img)

	def __getitem__(self, index):
		image_file = self.image_files[index]
		image = Image.open(image_file).convert('RGB')
		image = self.image_transform(image)

		if self.is_train:
			return image
		else:
			if os.path.dirname(image_file).endswith("good"):
				target = torch.zeros([1, image.shape[-2], image.shape[-1]])
			else:
				target = Image.open(
					image_file.replace("test", "ground_truth").replace(
						".png", "_mask.png"
					)
				)
				target = self.target_transform(target)
			return image, target, image_file

	def __len__(self):
		return len(self.image_files)


def mvtec_un_normalize(torch_img):
	un_normalize_transform = transforms.Normalize((-MEAN / STD).tolist(), (1.0 / STD).tolist())
	return un_normalize_transform(torch_img)


def get_debug_images_paths(category):
	debug_images_paths = []
	debug_images = yaml.safe_load(open(str(Path(__file__).resolve().parent / "mvtec_debug_images.yaml"), "r"))
	for subdir in debug_images[category]['test'].items():
		for img_number in subdir[1]:
			debug_images_paths.append(str(Path(category) / "test" / subdir[0] / f"{img_number:03d}.png"))
	return debug_images_paths
