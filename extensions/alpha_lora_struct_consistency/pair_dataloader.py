import copy
import math
import os
import random
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

from torch.utils.data import BatchSampler, DataLoader, Dataset

from toolkit.accelerator import get_accelerator
from toolkit.data_loader import get_dataloader_datasets, is_macos, is_native_windows
from toolkit.data_transfer_object.data_loader import DataLoaderBatchDTO, FileItemDTO
from toolkit.print import print_acc


def _sample_key_from_item(subdataset, file_item: FileItemDTO) -> str:
    dataset_root = subdataset.dataset_path
    if not os.path.isdir(dataset_root):
        dataset_root = os.path.dirname(dataset_root)
    rel_path = os.path.relpath(file_item.path, dataset_root)
    stem, _ = os.path.splitext(rel_path)
    return stem.replace("\\", "/")


def _copy_spatial_params(dst: FileItemDTO, src: FileItemDTO) -> None:
    for attr in (
        "scale_to_width",
        "scale_to_height",
        "crop_x",
        "crop_y",
        "crop_width",
        "crop_height",
        "flip_x",
        "flip_y",
    ):
        setattr(dst, attr, copy.deepcopy(getattr(src, attr)))


class SameSampleStrengthPairDataset(Dataset):
    """
    Wrap existing AiToolkitDataset instances and emit two views of the same
    underlying sample at different strength labels.

    Each __getitem__ returns a dict with:
      - items: [FileItemDTO_a, FileItemDTO_b]
      - group_key: shared sample id
    """

    def __init__(
        self,
        datasets: Sequence,
        share_spatial_crop: bool = True,
        seed: int = 42,
    ):
        super().__init__()
        self.datasets = list(datasets)
        self.share_spatial_crop = bool(share_spatial_crop)
        self.seed = int(seed)
        self.epoch_num = 0
        self.current_epoch_seed = self.seed
        self.group_to_members: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        self.bucket_to_indices: Dict[Tuple[int, int], List[int]] = {}

        for ds_idx, dataset in enumerate(self.datasets):
            for item_idx, file_item in enumerate(dataset.file_list):
                key = _sample_key_from_item(dataset, file_item)
                self.group_to_members[key].append((ds_idx, item_idx))

        self.group_keys = sorted(
            [key for key, members in self.group_to_members.items() if len(members) >= 2]
        )
        if len(self.group_keys) == 0:
            raise ValueError(
                "No same-sample multi-strength groups found for structural consistency training."
            )
        self.setup_epoch()

    def setup_epoch(self):
        for dataset in self.datasets:
            if hasattr(dataset, "setup_epoch"):
                dataset.setup_epoch()
                if hasattr(dataset, "len"):
                    dataset.len = None
        self.current_epoch_seed = self.seed + self.epoch_num * 1000003
        rng = random.Random(self.current_epoch_seed)
        groups_by_bucket: Dict[Tuple[int, int], List[str]] = defaultdict(list)
        for key in self.group_keys:
            members = self.group_to_members[key]
            bucket_key = None
            for ds_idx, item_idx in members:
                file_item = self.datasets[ds_idx].file_list[item_idx]
                crop_width = int(getattr(file_item, "crop_width", 0) or 0)
                crop_height = int(getattr(file_item, "crop_height", 0) or 0)
                if crop_width > 0 and crop_height > 0:
                    bucket_key = (crop_width, crop_height)
                    break
            if bucket_key is None:
                bucket_key = (0, 0)
            groups_by_bucket[bucket_key].append(key)

        shuffled_group_keys: List[str] = []
        bucket_items = list(groups_by_bucket.items())
        rng.shuffle(bucket_items)
        for _, bucket_group_keys in bucket_items:
            bucket_group_keys = list(bucket_group_keys)
            rng.shuffle(bucket_group_keys)
            shuffled_group_keys.extend(bucket_group_keys)

        self.group_keys = shuffled_group_keys
        index_by_key = {key: idx for idx, key in enumerate(self.group_keys)}
        self.bucket_to_indices = {
            bucket_key: [index_by_key[key] for key in bucket_group_keys]
            for bucket_key, bucket_group_keys in groups_by_bucket.items()
        }
        self.epoch_num += 1

    def __len__(self):
        return len(self.group_keys)

    def _materialize_item(
        self,
        dataset_idx: int,
        item_idx: int,
        shared_anchor: FileItemDTO = None,
    ) -> FileItemDTO:
        dataset = self.datasets[dataset_idx]
        file_item = copy.deepcopy(dataset.file_list[item_idx])
        if shared_anchor is not None and self.share_spatial_crop:
            _copy_spatial_params(file_item, shared_anchor)
        file_item.load_and_process_image(dataset.transform)
        file_item.load_caption(dataset.caption_dict)
        return file_item

    def __getitem__(self, index):
        group_key = self.group_keys[index]
        members = self.group_to_members[group_key]
        rng = random.Random(self.seed + self.epoch_num * 1000003 + index * 9176)
        first_member, second_member = rng.sample(members, 2)

        first_dataset = self.datasets[first_member[0]]
        second_dataset = self.datasets[second_member[0]]
        first_anchor = copy.deepcopy(first_dataset.file_list[first_member[1]])
        second_anchor = copy.deepcopy(second_dataset.file_list[second_member[1]])
        shared_anchor = first_anchor if rng.random() < 0.5 else second_anchor

        item_a = self._materialize_item(
            first_member[0],
            first_member[1],
            shared_anchor=shared_anchor,
        )
        item_b = self._materialize_item(
            second_member[0],
            second_member[1],
            shared_anchor=shared_anchor,
        )

        return {
            "items": [item_a, item_b],
            "group_key": group_key,
        }


class _PairBucketBatchSampler(BatchSampler):
    def __init__(
        self,
        dataset: SameSampleStrengthPairDataset,
        batch_size: int,
        drop_last: bool = False,
        num_replicas: int = 1,
        rank: int = 0,
        seed: int = 42,
    ):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.num_replicas = max(1, int(num_replicas))
        self.rank = int(rank)
        self.seed = int(seed)

    def __iter__(self):
        rng = random.Random(self.seed + self.dataset.current_epoch_seed + 17)
        bucket_items = list(self.dataset.bucket_to_indices.items())
        rng.shuffle(bucket_items)

        batches: List[List[int]] = []
        for _, indices in bucket_items:
            bucket_indices = list(indices)
            rng.shuffle(bucket_indices)
            for start in range(0, len(bucket_indices), self.batch_size):
                batch = bucket_indices[start : start + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)

        rng.shuffle(batches)
        if len(batches) == 0:
            return iter(())

        if self.num_replicas > 1:
            if self.drop_last:
                total_batches = (len(batches) // self.num_replicas) * self.num_replicas
                batches = batches[:total_batches]
            else:
                remainder = len(batches) % self.num_replicas
                if remainder != 0:
                    pad = self.num_replicas - remainder
                    batches.extend(
                        copy.deepcopy(batches[i % len(batches)]) for i in range(pad)
                    )
            batches = batches[self.rank :: self.num_replicas]
        return iter(batches)

    def __len__(self):
        batch_count = 0
        for indices in self.dataset.bucket_to_indices.values():
            if self.drop_last:
                batch_count += len(indices) // self.batch_size
            else:
                batch_count += math.ceil(len(indices) / self.batch_size)
        if self.num_replicas <= 1:
            return batch_count
        if self.drop_last:
            return batch_count // self.num_replicas
        return math.ceil(batch_count / self.num_replicas)


def _pair_collate_fn(pair_batch: List[dict]) -> DataLoaderBatchDTO:
    file_items: List[FileItemDTO] = []
    consistency_pairs: List[Tuple[int, int]] = []
    consistency_group_keys: List[str] = []
    consistency_group_ids: List[int] = []

    for group_idx, pair_entry in enumerate(pair_batch):
        start_idx = len(file_items)
        pair_items = pair_entry["items"]
        if len(pair_items) != 2:
            raise ValueError(
                f"Expected exactly 2 items per same-sample pair, got {len(pair_items)}"
            )
        file_items.extend(pair_items)
        consistency_pairs.append((start_idx, start_idx + 1))
        consistency_group_keys.extend([pair_entry["group_key"], pair_entry["group_key"]])
        consistency_group_ids.extend([group_idx, group_idx])

    batch = DataLoaderBatchDTO(file_items=file_items)
    batch.consistency_pairs = consistency_pairs
    batch.consistency_group_keys = consistency_group_keys
    batch.consistency_group_ids = consistency_group_ids
    return batch


def build_same_sample_pair_dataloader_from_existing(
    base_dataloader,
    batch_size: int,
    share_spatial_crop: bool = True,
    seed: int = 42,
):
    if batch_size % 2 != 0:
        raise ValueError(
            f"Structural consistency pair loader requires an even batch size, got {batch_size}"
        )

    subdatasets = get_dataloader_datasets(base_dataloader)
    pair_dataset = SameSampleStrengthPairDataset(
        subdatasets,
        share_spatial_crop=share_spatial_crop,
        seed=seed,
    )

    pair_batch_size = max(1, batch_size // 2)
    config0 = subdatasets[0].dataset_config
    dataloader_kwargs = {}
    if is_native_windows() or is_macos():
        dataloader_kwargs["num_workers"] = 0
    else:
        dataloader_kwargs["num_workers"] = config0.num_workers
        if config0.num_workers > 0:
            dataloader_kwargs["prefetch_factor"] = config0.prefetch_factor

    accelerator = get_accelerator()
    batch_sampler = _PairBucketBatchSampler(
        pair_dataset,
        batch_size=pair_batch_size,
        drop_last=False,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        seed=seed,
    )

    print_acc(
        "[alpha_lora_struct_consistency] Replacing standard loader with "
        f"same-sample pair loader: {len(pair_dataset.group_keys)} grouped samples, "
        f"item-batch={batch_size}, pair-batch={pair_batch_size}"
    )

    return DataLoader(
        pair_dataset,
        batch_sampler=batch_sampler,
        collate_fn=_pair_collate_fn,
        **dataloader_kwargs,
    )
