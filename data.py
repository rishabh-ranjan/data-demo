import os
import json

import line_profiler
import torch
from data_rs import Dataset as RustDataset
from torch.nn.attention.flex_attention import create_block_mask, create_mask
from torch.utils.data import Dataset, IterableDataset


class FineTuneDataset(Dataset):
    def __init__(
        self,
        ctx_len,
        mask_prob,
        dataset_name,
        task_name,
        task_split,
        dropout_tok,
        dropout_f2p,
        dropout_p2f,
    ):
        self.dataset = RustDataset(
            dataset_name=dataset_name,
            ctx_len=ctx_len,
            mask_prob=mask_prob,
            dropout_tok=dropout_tok,
            dropout_f2p=dropout_f2p,
            dropout_p2f=dropout_p2f,
        )

        table_info_path = (
            f"{os.environ['HOME']}/scratch/pre/{dataset_name}/table_info.json"
        )
        with open(table_info_path) as f:
            table_info = json.load(f)

        if task_split == "train":
            task_split = "Train"
        elif task_split == "val":
            task_split = "Val"
        elif task_split == "test":
            task_split = "Test"

        self.info = table_info[f"{task_name}:{task_split}"]

        emb_path = f"{os.environ['HOME']}/scratch/pre/{dataset_name}/text_emb.pt"
        self.id_to_emb = torch.load(emb_path, map_location="cpu", mmap=True)

    def __len__(self):
        return self.info["num_nodes"]

    @line_profiler.profile
    def __getitem__(self, idx):
        assert idx < self.info["num_nodes"]
        seed_node_idx = self.info["node_idx_offset"] + idx
        out = dict(self.dataset.get_item_py(seed_node_idx))
        out = {k: torch.from_numpy(v) for k, v in out.items()}
        out["f2p_nbr_idxs"] = out["f2p_nbr_idxs"].view(-1, 4)
        out["number_values"] = out["number_values"].view(-1, 1)
        out["datetime_values"] = out["datetime_values"].view(-1, 1)
        out["table_name_idxs"] = out["table_name_values"]
        for k in [
            "text_values",
            "table_name_values",
            "col_name_values",
        ]:
            out[k] = self.id_to_emb[out[k].long()]
        for k in [
            "number_values",
            "datetime_values",
            "text_values",
            "table_name_values",
            "col_name_values",
        ]:
            out[k] = out[k].to(torch.bfloat16)
        return out
