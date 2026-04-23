import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class MovingMNISTDataset(Dataset):
    """
    Moving MNIST dataset for video prediction / world model training.

    Expected raw data shape:
        1) [N, T, H, W]
        2) [N, T, 1, H, W]

    Returned dict:
        {
            "video":   [T, C, H, W],
            "context": [K, C, H, W],
            "target":  [H_pred, C, H, W],
        }
    """

    def __init__(
        self,
        data_path: str,
        seq_len: int = 20,
        context_len: int = 10,
        pred_len: int = 10,
        image_size: int = 64,
    ):
        super().__init__()

        assert os.path.exists(data_path), f"Data file not found: {data_path}"
        assert seq_len == context_len + pred_len, (
            f"seq_len must equal context_len + pred_len, "
            f"but got seq_len={seq_len}, context_len={context_len}, pred_len={pred_len}"
        )

        self.data_path = data_path
        self.seq_len = seq_len
        self.context_len = context_len
        self.pred_len = pred_len
        self.image_size = image_size

        self.data = np.load(data_path)

        if self.data.ndim not in (4, 5):
            raise ValueError(
                f"Expected data ndim to be 4 or 5, but got shape {self.data.shape}"
            )

        if self.data.shape[1] < self.seq_len:
            raise ValueError(
                f"Dataset sequence length {self.data.shape[1]} is smaller than required seq_len={self.seq_len}"
            )

    def __len__(self):
        return len(self.data)

    def _to_tensor_video(self, video_np: np.ndarray) -> torch.Tensor:
        """
        Convert raw numpy video to torch float tensor with shape [T, 1, H, W],
        normalized to [0, 1].
        """
        video = torch.from_numpy(video_np).float()

        # Case 1: [T, H, W] -> [T, 1, H, W]
        if video.ndim == 3:
            video = video.unsqueeze(1)

        # Case 2: already [T, 1, H, W]
        elif video.ndim == 4:
            pass

        else:
            raise ValueError(f"Unexpected single video shape: {video.shape}")

        # Normalize to [0, 1]
        if video.max() > 1.0:
            video = video / 255.0

        # Resize if needed
        _, c, h, w = video.shape
        if h != self.image_size or w != self.image_size:
            video = F.interpolate(
                video,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )

        return video

    def __getitem__(self, idx):
        """
        Return:
            video:   [T, 1, H, W]
            context: [K, 1, H, W]
            target:  [pred_len, 1, H, W]
        """
        video_np = self.data[idx]  # [T,H,W] or [T,1,H,W]

        # truncate sequence to seq_len
        video_np = video_np[: self.seq_len]

        video = self._to_tensor_video(video_np)

        context = video[: self.context_len]
        target = video[self.context_len : self.context_len + self.pred_len]

        sample = {
            "video": video,         # [T, 1, H, W]
            "context": context,     # [K, 1, H, W]
            "target": target,       # [pred_len, 1, H, W]
        }
        return sample


def build_dataloader(
    data_path: str,
    seq_len: int,
    context_len: int,
    pred_len: int,
    image_size: int,
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
    num_workers: int,
    pin_memory: bool = False,
):
    dataset = MovingMNISTDataset(
        data_path=data_path,
        seq_len=seq_len,
        context_len=context_len,
        pred_len=pred_len,
        image_size=image_size,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return dataset, loader