import os
import sys
import torch
import numpy as np
import pandas as pd
from tqdm import trange
from util import hss, tss, conf_mat
from discriminator_gan import Discriminator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data_v2_from_npy(x_path, label_path):
    x = np.load(x_path)
    y = np.load(label_path)
    return x, y


def load_data_v1_from_npy(x_path, label_path):
    x = np.load(x_path)
    label_string = list(np.load(label_path))

    def string_label_to_int(label_string):
        if (
            label_string.startswith("FQ")
            or label_string.startswith("B")
            or label_string.startswith("C")
        ):
            return 1
        elif label_string.startswith("M") or label_string.startswith("X"):
            return 0
        else:
            raise KeyError

    label_int = np.array(list(map(string_label_to_int, label_string)))
    return x[:, :, [0, 1, 4, 5]], label_int


def load_data_from_csv(partition_path):
    x, y = [], []
    for csv_file in os.listdir(os.path.join(partition_path, "FL"))[:10]:
        csv_path = os.path.join(partition_path, "FL", csv_file)
        instance_df = pd.read_csv(csv_path, sep="\t")
        TOTUSJH = instance_df["TOTUSJH"].to_numpy()
        TOTBSQ = instance_df["TOTBSQ"].to_numpy()
        ABSNJZH = instance_df["ABSNJZH"].to_numpy()
        SAVNCPP = instance_df["SAVNCPP"].to_numpy()
        instance = np.column_stack((TOTUSJH, TOTBSQ, ABSNJZH, SAVNCPP))
        x.append(instance)
        y.append(0)
    for csv_file in os.listdir(os.path.join(partition_path, "NF"))[:10]:
        csv_path = os.path.join(partition_path, "NF", csv_file)
        instance_df = pd.read_csv(csv_path, sep="\t")
        TOTUSJH = instance_df["TOTUSJH"].to_numpy()
        TOTBSQ = instance_df["TOTBSQ"].to_numpy()
        ABSNJZH = instance_df["ABSNJZH"].to_numpy()
        SAVNCPP = instance_df["SAVNCPP"].to_numpy()
        instance = np.column_stack((TOTUSJH, TOTBSQ, ABSNJZH, SAVNCPP))
        x.append(instance)
        y.append(1)
    return np.array(x), np.array(y)


def main():
    partition, is_normalized = sys.argv[1], sys.argv[2]
    
    model = Discriminator(hidden_dim=64, seq_len=60, n_feat=4).to(device)
    model.load_state_dict(
        torch.load("./../discriminator_epoch-80.pth", map_location=device, weights_only=True)
    )

    batch_size = 64

    if not bool(is_normalized):
        print('Normalizing the data')
        min_max = get_reference_min_max("./../p1_x_raw.npy", "./../p1_label.npy")
        x, y = load_data_v1_from_npy(f"./../p{partition}_x_raw.npy", f"./../p{partition}_label.npy")
        for i in range(4):
            min_, max_ = min_max[i]
            x[:, :, i] = (x[:, :, i] - min_) / (max_ - min_)
    else:
        print('Data is already normalized')
        x, y = load_data_v2_from_npy(f"./../normalized/p{partition}_x.npy", f"./../normalized/p{partition}_y.npy")

    x = torch.tensor(x, dtype=torch.float32).to(device)

    tp, fp, tn, fn = 0, 0, 0, 0
    model.eval()
    with torch.no_grad():
        for i in trange(0, x.shape[0], batch_size):
            y_hat = model(x[i : i + batch_size])
            y_hat[y_hat < 0.5] = 0
            y_hat[y_hat >= 0.5] = 1
            tp_, fp_, tn_, fn_ = conf_mat(y_hat, y[i : i + batch_size])
            tp += tp_
            fp += fp_
            tn += tn_
            fn += fn_

    hss_score = hss(tp, fp, tn, fn)
    tss_score = tss(tp, fp, tn, fn)

    print('partition', partition)
    print("hss", hss_score)
    print("tss", tss_score)
    print("tp, fp, tn, fn", tp, fp, tn, fn)


def get_reference_min_max(partition_1_x_path, partition_1_label_path):
    x, _ = load_data_from_npy(partition_1_x_path, partition_1_label_path)
    min_max = [
        [
            np.min(x[:, :, 0], axis=(0, 1)),
            np.max(x[:, :, 0], axis=(0, 1)),
        ],
        [
            np.min(x[:, :, 1], axis=(0, 1)),
            np.max(x[:, :, 1], axis=(0, 1)),
        ],
        [
            np.min(x[:, :, 2], axis=(0, 1)),
            np.max(x[:, :, 2], axis=(0, 1)),
        ],
        [
            np.min(x[:, :, 3], axis=(0, 1)),
            np.max(x[:, :, 3], axis=(0, 1)),
        ],
    ]
    return min_max


if __name__ == "__main__":
    main()
