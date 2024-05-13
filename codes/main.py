import math
import os.path
import time
import warnings
from typing import Dict, Literal

import pandas as pd

warnings.simplefilter("ignore")
import numpy as np
import scipy.special
import sklearn.datasets
import sklearn.metrics
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import sklearn.model_selection
import sklearn.preprocessing
import torch
import torch.nn.functional as F
import torch.optim
from torch import Tensor
from tqdm.std import tqdm
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler
import delu
import shap

warnings.resetwarnings()
from rtdl_revisiting_models import MLP, ResNet, FTTransformer

# *******************************************
# B1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device('mps')
# Set random seeds in all libraries.
delu.random.seed(0)

# *******************************************
# B2
TaskType = Literal["regression", "binclass", "multiclass"]
# task_type: TaskType = "regression"
# n_classes = None
# dataset = sklearn.datasets.fetch_california_housing()
# X_cont: np.ndarray = dataset["data"]
# Y: np.ndarray = dataset["target"]

# NOTE: uncomment to solve a classification task.
# dataset = sklearn.datasets.load_breast_cancer()
n_classes = 7  # 7 @ ginseng , 3@zj
assert n_classes >= 2
task_type: TaskType = 'binclass' if n_classes == 2 else 'multiclass'

# ori code
# X_cont, Y = sklearn.datasets.make_classification(
#     n_samples=20000,
#     n_features=8,
#     n_classes=n_classes,
#     n_informative=3,
#     n_redundant=2,
# )

# *** add code ****
csv_path = '/project/2024/Pan-0507/data/train_data_01.csv'

assert os.path.exists(csv_path)
frames = pd.read_csv(csv_path)
X_cont = frames.values[:, 1:]
Y = frames.values[:, 0].astype(np.int64)

# ros = RandomOverSampler(random_state=42, sampling_strategy=0.1)
# X_cont, Y = ros.fit_resample(X_cont, Y)
#
# rus = RandomUnderSampler(random_state=42, sampling_strategy=0.3)
# X_cont, Y = rus.fit_resample(X_cont, Y)

scaler = StandardScaler()
# X_cont[:, 1:] = scaler.fit_transform(X_cont[:, 1:])
X_cont = scaler.fit_transform(X_cont)

# *********************************************

# >>> Continuous features.
X_cont: np.ndarray = X_cont.astype(np.float32)
n_cont_features = X_cont.shape[1]

# >>> Categorical features.
# NOTE: the above datasets do not have categorical features, but,
# for the demonstration purposes, it is possible to generate them.
cat_cardinalities = [
    # NOTE: uncomment the two lines below to add two categorical features.
    # 4,  # Allowed values: [0, 1, 2, 3].
    # 7,  # Allowed values: [0, 1, 2, 3, 4, 5, 6].
]
X_cat = (
    np.column_stack(
        [np.random.randint(0, c, (len(X_cont),)) for c in cat_cardinalities]
    )
    if cat_cardinalities
    else None
)

# >>> Labels.
# Regression labels must be represented by float32.
if task_type == "regression":
    Y = Y.astype(np.float32)
else:
    assert n_classes is not None
    Y = Y.astype(np.int64)
    assert set(Y.tolist()) == set(
        range(n_classes)
    ), "Classification labels must form the range [0, 1, ..., n_classes - 1]"

# >>> Split the dataset.
all_idx = np.arange(len(Y))
trainval_idx, test_idx = sklearn.model_selection.train_test_split(
    all_idx, train_size=0.8
)
train_idx, val_idx = sklearn.model_selection.train_test_split(
    trainval_idx, train_size=0.8
)
data_numpy = {
    "train": {"x_cont": X_cont[train_idx], "y": Y[train_idx]},
    "val": {"x_cont": X_cont[val_idx], "y": Y[val_idx]},
    "test": {"x_cont": X_cont[test_idx], "y": Y[test_idx]},
}
if X_cat is not None:
    data_numpy["train"]["x_cat"] = X_cat[train_idx]
    data_numpy["val"]["x_cat"] = X_cat[val_idx]
    data_numpy["test"]["x_cat"] = X_cat[test_idx]


X_cont_train_numpy = data_numpy["train"]["x_cont"]
noise = (
    np.random.default_rng(0)
    .normal(0.0, 1e-5, X_cont_train_numpy.shape)
    .astype(X_cont_train_numpy.dtype)
)
preprocessing = sklearn.preprocessing.QuantileTransformer(
    n_quantiles=max(min(len(train_idx) // 30, 1000), 10),
    output_distribution="normal",
    subsample=10 ** 9,
).fit(X_cont_train_numpy + noise)
del X_cont_train_numpy

for part in data_numpy:
    data_numpy[part]["x_cont"] = preprocessing.transform(data_numpy[part]["x_cont"])

# >>> Label preprocessing.
if task_type == "regression":
    Y_mean = data_numpy["train"]["y"].mean().item()
    Y_std = data_numpy["train"]["y"].std().item()
    for part in data_numpy:
        data_numpy[part]["y"] = (data_numpy[part]["y"] - Y_mean) / Y_std

# >>> Convert data to tensors.
data = {
    part: {k: torch.as_tensor(v, device=device) for k, v in data_numpy[part].items()}
    for part in data_numpy
}

if task_type != "multiclass":
    # Required by F.binary_cross_entropy_with_logits
    for part in data:
        data[part]["y"] = data[part]["y"].float()

# ****************************************
# B4
# The output size.
d_out = n_classes if task_type == "multiclass" else 1


# NOTE: uncomment to train ResNet
model = ResNet(
    d_in=n_cont_features + sum(cat_cardinalities),
    d_out=d_out,
    n_blocks=2,
    d_block=192,
    d_hidden=None,
    d_hidden_multiplier=2.0,
    dropout1=0.3,
    dropout2=0.0,
).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)



# *******************************
# B5
def apply_model(batch: Dict[str, Tensor]) -> Tensor:
    if isinstance(model, (MLP, ResNet)):
        x_cat_ohe = (
            [
                F.one_hot(column, cardinality)
                for column, cardinality in zip(batch["x_cat"].T, cat_cardinalities)
            ]
            if "x_cat" in batch
            else []
        )
        return model(torch.column_stack([batch["x_cont"]] + x_cat_ohe)).squeeze(-1)

    elif isinstance(model, FTTransformer):
        return model(batch["x_cont"], batch.get("x_cat")).squeeze(-1)

    else:
        raise RuntimeError(f"Unknown model type: {type(model)}")


loss_fn = (
    F.binary_cross_entropy_with_logits
    if task_type == "binclass"
    else F.cross_entropy
    if task_type == "multiclass"
    else F.mse_loss
)


@torch.no_grad()
def evaluate(part: str) -> float:
    model.eval()

    eval_batch_size = 8096
    y_pred = (
        torch.cat(
            [
                apply_model(batch)
                for batch in delu.iter_batches(data[part], eval_batch_size)
            ]
        )
        .cpu()
        .numpy()
    )
    y_true = data[part]["y"].cpu().numpy()

    if task_type == "binclass":
        y_pred = np.round(scipy.special.expit(y_pred))
        score = sklearn.metrics.accuracy_score(y_true, y_pred)
    elif task_type == "multiclass":
        y_pred = y_pred.argmax(1)
        score_acc = accuracy_score(y_true, y_pred)
        score_prec = precision_score(y_true, y_pred, average='macro')
        score_rec = recall_score(y_true, y_pred, average='macro')
        score_f1 = f1_score(y_true, y_pred, average='macro')
        # score_auc = sklearn.metrics.roc_auc_score(y_true, y_pred)

        score = {'acc': score_acc, 'prec': score_prec, 'rec': score_rec,
                 'f1': score_f1}


    else:
        assert task_type == "regression"
        score = -(sklearn.metrics.mean_squared_error(y_true, y_pred) ** 0.5 * Y_std)
    return score  # The higher -- the better.


def compute_kl_loss(p, q, pad_mask=None):
    p_log_soft = F.log_softmax(p, dim=-1)
    p_soft = F.softmax(p, dim=-1)

    q_log_soft = F.log_softmax(q, dim=-1)
    q_soft = F.softmax(q, dim=-1)

    p_loss = F.kl_div(p_log_soft, p_soft, reduction='none')
    q_loss = F.kl_div(q_log_soft, q_soft, reduction='none')

    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        p_loss.masked_fill_(pad_mask, 0.)

    p_loss = p_loss.sum()
    q_loss = q_loss.sum()

    kl_loss = (p_loss + q_loss) * 0.5
    return kl_loss


# print(f'Test score before training: {evaluate("test"):.4f}')

# ***********************************
# B 6 Training
# For demonstration purposes (fast training and bad performance),
# one can set smaller values:
# n_epochs = 20
# patience = 2
n_epochs = 1_000_000_000
patience = 32

batch_size = 128  # 256
epoch_size = math.ceil(len(train_idx) / batch_size)
timer = delu.tools.Timer()
early_stopping = delu.tools.EarlyStopping(patience, mode="max")
best = {
    "val": -math.inf,
    "test": -math.inf,
    "epoch": -1,
}

print(f"Device: {device.type.upper()}")
print("-" * 88 + "\n")
timer.run()
for epoch in range(n_epochs):
    for batch in tqdm(
            delu.iter_batches(data["train"], batch_size, shuffle=True),
            desc=f"Epoch {epoch}",
            total=epoch_size,
    ):
        model.train()
        optimizer.zero_grad()
        # loss = loss_fn(apply_model(batch), batch["y"])
        model_pred = apply_model(batch)
        model_pred_2 = apply_model(batch)

        loss_bce = loss_fn(model_pred, batch['y'])
        kl_loss = compute_kl_loss(model_pred, model_pred_2)
        #
        loss = loss_bce + 1 * kl_loss
        # print('loss {}'.format(loss.data), 'kl_loss {}'.format(kl_loss.data))
        # loss = loss_bce

        # loss R drop

        loss.backward()
        optimizer.step()

    val_score = evaluate("val")
    test_score = evaluate("test")
    if isinstance(val_score, float):
        print(f"(val) {val_score:.4f} (test) {test_score:.4f} [time] {timer}")

        early_stopping.update(val_score)
        if early_stopping.should_stop():
            break

        if val_score > best["val"]:
            print("🌸 New best epoch! 🌸")
            best = {"val": val_score, "test": test_score, "epoch": epoch}

    else:
        print(f"(val) {val_score['acc']:.4f} {val_score['prec']:.4f} {val_score['rec']:.4f} "
              f"{val_score['f1']:.4f}"
              f"(test) {test_score['acc']:.4f} {test_score['prec']:.4f} {test_score['rec']:.4f} "
              f"{test_score['f1']:.4f} [time] {timer}")

        early_stopping.update(val_score['acc'])
        if early_stopping.should_stop():
            break

        if test_score['acc'] > best["test"]:
            print("🌸 New best epoch! 🌸")
            best = {"val": val_score['acc'], "test": test_score['acc'], "epoch": epoch}

    print()


