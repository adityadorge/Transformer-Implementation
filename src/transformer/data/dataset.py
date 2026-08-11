from pathlib import Path
from typing import Tuple

import torch


class GPTDataBatcher:
    """Loads preprocessed token arrays from disk.
    Serves sliding window inputs and targets.
    """

    def __init__(self, tensor_path: Path, block_size: int, batch_size: int):
        self.block_size = block_size
        self.batch_size = batch_size

        if not tensor_path.exists():
            raise FileNotFoundError(
                f"Processed tensor target binary file missing: {tensor_path}"
            )

        # Fast initialization: load memory-mapped binaries into active memory
        self.data_tensor = torch.load(tensor_path, weights_only=True)

        # Protect context boundaries from sliding overflows
        self.max_start_index = len(self.data_tensor) - self.block_size - 1
        if self.max_start_index <= 0:
            raise ValueError(
                "Tensor token array is too small to fulfill block_size context."
            )

    def get_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Samples random offsets to build parallel matrix batches.
        Returns inputs (X) and shifted targets (Y).
        """
        random_offsets = torch.randint(
            high=self.max_start_index, size=(self.batch_size,)
        )

        # Parallel extraction matching your sequence design
        x = torch.stack(
            [self.data_tensor[i : i + self.block_size] for i in random_offsets]
        )
        y = torch.stack(
            [self.data_tensor[i + 1 : i + self.block_size + 1] for i in random_offsets]
        )

        return x, y
