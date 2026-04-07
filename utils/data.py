import io
import math
import random

import h5py
import pandas as pd
import torch
import torch.distributed as dist
from PIL import Image, ImageFile
from torch.utils.data import Dataset, Sampler

from utils.constant import FAMILY, GENUS, NOT_CLASSIFIED, ORDER, SPECIES

ImageFile.LOAD_TRUNCATED_IMAGES = True


def convert_uri_to_index_list(uri_list):
    string_to_int = {}
    next_int = 0
    integers = []
    for value in uri_list:
        if value not in string_to_int:
            string_to_int[value] = next_int
            next_int += 1
        integers.append(string_to_int[value])
    return integers


class CsvDatasetText(Dataset):
    def __init__(self, input_filename, caption_key, text_tokenizer):
        df = pd.read_csv(input_filename)
        self.unique_captions = list(set(df[caption_key].tolist()))
        self.text_tokenize = text_tokenizer

    def __len__(self):
        return len(self.unique_captions)

    def __getitem__(self, idx):
        return self.text_tokenize([str(self.unique_captions[idx])])[0]


class CsvDataset(Dataset):
    def __init__(
        self,
        input_filename,
        image_filename,
        img_transforms,
        img_key,
        caption_key,
        dna_key,
        text_tokenizer,
    ):
        df = pd.read_csv(input_filename)
        self.hdf5_images = h5py.File(image_filename, "r")["bioscan_dataset"]
        self.images_id = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        self.barcodes = df[dna_key].tolist()
        self.transforms = img_transforms
        self.text_tokenize = text_tokenizer

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        image = Image.open(io.BytesIO(self.hdf5_images[str(self.images_id[idx])][:]))
        image = self.transforms(image)
        caption_text = self.captions[idx]
        caption_tokens = self.text_tokenize([str(caption_text)])[0]
        dna_text = self.barcodes[idx]
        return image, caption_tokens, caption_text, dna_text


class BIOSCANHierarchicalDataset(Dataset):
    def __init__(
        self,
        input_filename,
        image_filename,
        img_transforms,
        img_key,
        caption_key,
        dna_key,
        text_tokenizer,
        use_text_label=True,
    ):
        df = pd.read_csv(input_filename)
        self.hdf5_images = h5py.File(image_filename, "r")["bioscan_dataset"]
        self.images_id = df[img_key].tolist()
        self.captions = df[caption_key].tolist()
        self.barcodes = df[dna_key].tolist()
        self.labels = df["unique_label"].tolist() if use_text_label else list(range(len(df)))
        self.transforms = img_transforms
        self.text_tokenize = text_tokenizer

        self.filenames = []
        self.hierarchical_labels = {ORDER: {}, FAMILY: {}, GENUS: {}, SPECIES: {}}
        self.all_labels = {}

        for row_index, row in df.iterrows():
            filename = row["image_file"]
            order = row["order"]
            family = row["family"]
            genus = row["genus"]
            species = row["species"]
            self.all_labels[filename] = eval(row["All_level_label"])
            self.filenames.append(filename)

            if order != NOT_CLASSIFIED:
                self.hierarchical_labels[ORDER].setdefault(order, {})[filename] = row_index
            if family != NOT_CLASSIFIED:
                self.hierarchical_labels[FAMILY].setdefault(family, {})[filename] = row_index
            if genus != NOT_CLASSIFIED:
                self.hierarchical_labels[GENUS].setdefault(genus, {})[filename] = row_index
            if species != NOT_CLASSIFIED:
                self.hierarchical_labels[SPECIES].setdefault(species, {})[filename] = row_index

    def get_label_split_by_index(self, index):
        filename = self.filenames[index]
        return self.all_labels[filename] + [index]

    def __getitem__(self, batch_indices):
        images0_batch = []
        images1_batch = []
        hierarchical_labels_batch = []
        texts_batch = []
        barcodes_batch = []
        label_index_batch = []

        for idx in batch_indices:
            image = Image.open(io.BytesIO(self.hdf5_images[str(self.images_id[idx])][:]))
            image0, image1 = self.transforms(image)

            images0_batch.append(image0)
            images1_batch.append(image1)
            hierarchical_labels_batch.append(list(self.get_label_split_by_index(idx)))
            texts_batch.append(self.text_tokenize([str(self.captions[idx])])[0])
            barcodes_batch.append(self.barcodes[idx])
            label_index_batch.append(self.labels[idx])

        return (
            [torch.stack(images0_batch), torch.stack(images1_batch)],
            torch.tensor(hierarchical_labels_batch),
            texts_batch,
            barcodes_batch,
            label_index_batch,
        )

    def random_sample(self, source_image, label_dict):
        current_dict = label_dict
        random_image = source_image
        if len(current_dict.keys()) != 1:
            while random_image == source_image:
                random_image = random.sample(list(current_dict.keys()), 1)[0]
        else:
            random_image = random.sample(list(current_dict.keys()), 1)[0]
        return current_dict[random_image]

    def __len__(self):
        return len(self.filenames)


class HierarchicalBatchSampler(Sampler):
    def __init__(self, batch_size, drop_last, dataset, num_replicas=None, rank=None):
        super().__init__(dataset)
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.dataset = dataset
        self.epoch = 0

        if num_replicas is None:
            num_replicas = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

        self.num_replicas = num_replicas
        self.rank = rank

        if self.drop_last and len(self.dataset) % self.num_replicas != 0:
            self.num_samples = math.ceil((len(self.dataset) - self.num_replicas) / self.num_replicas)
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def random_unvisited_sample(self, source_image, label, label_dict, visited, indices, remaining, num_attempt=10):
        if label in label_dict:
            attempt = 0
            while attempt < num_attempt:
                idx = self.dataset.random_sample(source_image, label_dict[label])
                if idx not in visited and idx in indices:
                    visited.add(idx)
                    return idx
                attempt += 1

        idx = remaining[torch.randint(len(remaining), (1,)).item()]
        visited.add(idx)
        return idx

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.epoch)
        batch = []
        visited = set()
        indices = torch.randperm(len(self.dataset), generator=generator).tolist()

        if not self.drop_last:
            indices += indices[: (self.total_size - len(indices))]
        else:
            indices = indices[: self.total_size]

        indices = indices[self.rank : self.total_size : self.num_replicas]
        remaining = list(set(indices).difference(visited))

        while len(remaining) > self.batch_size:
            idx = remaining[torch.randint(len(remaining), (1,)).item()]
            batch.append(idx)
            visited.add(idx)
            remaining = list(set(indices).difference(visited))

            order, family, genus, species, _ = self.dataset.get_label_split_by_index(idx)
            image_filename = self.dataset.filenames[idx]

            species_index = self.random_unvisited_sample(
                image_filename, species, self.dataset.hierarchical_labels[SPECIES], visited, indices, remaining
            )
            remaining = list(set(indices).difference(visited))
            genus_index = self.random_unvisited_sample(
                image_filename, genus, self.dataset.hierarchical_labels[GENUS], visited, indices, remaining
            )
            remaining = list(set(indices).difference(visited))
            family_index = self.random_unvisited_sample(
                image_filename, family, self.dataset.hierarchical_labels[FAMILY], visited, indices, remaining
            )
            remaining = list(set(indices).difference(visited))
            order_index = self.random_unvisited_sample(
                image_filename, order, self.dataset.hierarchical_labels[ORDER], visited, indices, remaining
            )

            batch.extend([species_index, genus_index, family_index, order_index])
            visited.update([species_index, genus_index, family_index, order_index])
            remaining = list(set(indices).difference(visited))

            if len(batch) >= self.batch_size:
                yield batch
                batch = []

        if remaining and not self.drop_last:
            batch.extend(list(remaining))
            yield batch

    def __len__(self):
        return self.num_samples // self.batch_size

    def set_epoch(self, epoch):
        self.epoch = epoch


BIOSCAN_HierarchihcalDataset = BIOSCANHierarchicalDataset
CsvDataset_Text = CsvDatasetText
