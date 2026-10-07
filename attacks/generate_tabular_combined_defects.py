# -*- coding: utf-8 -*-
"""Generate combined-defect assets for Bank, Adult and Default.

Scenarios
---------
FB  : fairness + backdoor
      The saved backdoor model retains the dataset/model's natural fairness gap
      and is additionally trained to respond to a deterministic tabular trigger.

FA  : fairness + adversarial
      No separate adversarial model is created. Strong multi-restart momentum CW-PGD samples are generated against
      the fixed clean model, which already exhibits the fairness defect.

FBA : fairness + backdoor + adversarial
      Strong multi-restart momentum CW-PGD samples are generated against the saved backdoor model. Therefore the
      evaluated system contains the fairness defect, the trained backdoor defect,
      and adversarial vulnerability at the same time.

Expected clean assets
---------------------
../data/fairness/<dataset>/x_train.h5, y_train.h5, g_train.h5,
                              x_test.h5,  y_test.h5,  g_test.h5,
                              metadata.json
../model/<dataset>.h5

Outputs
-------
../model/<dataset>_backdoor.h5
../data/combined_defects/<dataset>/fb/*
../data/combined_defects/<dataset>/fa/*
../data/combined_defects/<dataset>/fba/*
../data/combined_defects/<dataset>/manifest.json

The script uses only training/validation data to select the backdoor fine-tuning
length. Test data is used only for final reporting and attack asset generation.
"""

from __future__ import print_function

import argparse
import copy
import json
import os
import random
import shutil
from datetime import datetime

import h5py
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score


DATASETS = ("bank", "adult", "default")
EXPECTED_DIMS = {"bank": 32, "adult": 34, "default": 30}
SCENARIOS = ("fb", "fa", "fba")


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def read_h5(path, preferred_key):
    if not os.path.exists(path):
        raise FileNotFoundError("Missing H5 file: %s" % path)
    with h5py.File(path, "r") as handle:
        if preferred_key in handle:
            return np.asarray(handle[preferred_key])
        keys = list(handle.keys())
        if not keys:
            raise ValueError("Empty H5 file: %s" % path)
        for key in keys:
            obj = handle[key]
            if hasattr(obj, "shape"):
                return np.asarray(obj)
            if hasattr(obj, "keys"):
                for child_key in obj.keys():
                    child = obj[child_key]
                    if hasattr(child, "shape"):
                        return np.asarray(child)
    raise ValueError("Cannot locate an array in H5 file: %s" % path)


def write_h5(path, key, array):
    ensure_dir(os.path.dirname(path))
    with h5py.File(path, "w") as handle:
        handle.create_dataset(key, data=np.asarray(array), compression="gzip")


def load_json(path, required=False):
    if not os.path.exists(path):
        if required:
            raise FileNotFoundError("Missing JSON file: %s" % path)
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path, obj):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, ensure_ascii=False)


def load_dataset(data_root, dataset):
    dataset_dir = os.path.join(data_root, dataset)
    arrays = {
        "x_train": read_h5(os.path.join(dataset_dir, "x_train.h5"), "x"),
        "y_train": read_h5(os.path.join(dataset_dir, "y_train.h5"), "y"),
        "g_train": read_h5(os.path.join(dataset_dir, "g_train.h5"), "g"),
        "x_test": read_h5(os.path.join(dataset_dir, "x_test.h5"), "x"),
        "y_test": read_h5(os.path.join(dataset_dir, "y_test.h5"), "y"),
        "g_test": read_h5(os.path.join(dataset_dir, "g_test.h5"), "g"),
    }
    metadata = load_json(os.path.join(dataset_dir, "metadata.json"), required=True)

    arrays["x_train"] = np.asarray(arrays["x_train"], dtype=np.float32)
    arrays["x_test"] = np.asarray(arrays["x_test"], dtype=np.float32)
    for name in ("y_train", "g_train", "y_test", "g_test"):
        arrays[name] = np.asarray(arrays[name]).reshape(-1).astype(np.int64)

    if arrays["x_train"].ndim != 2:
        raise ValueError("%s x_train must be 2D" % dataset)
    if arrays["x_train"].shape[1] != EXPECTED_DIMS[dataset]:
        raise ValueError(
            "%s expected %d features, got %s"
            % (dataset, EXPECTED_DIMS[dataset], arrays["x_train"].shape)
        )
    if not (len(arrays["x_train"]) == len(arrays["y_train"]) == len(arrays["g_train"])):
        raise ValueError("%s train X/Y/G lengths differ" % dataset)
    if not (len(arrays["x_test"]) == len(arrays["y_test"]) == len(arrays["g_test"])):
        raise ValueError("%s test X/Y/G lengths differ" % dataset)
    return dataset_dir, arrays, metadata


def reconstruct_fit_validation(arrays, metadata):
    fit_count = metadata.get("fit_sample_count")
    val_count = metadata.get("validation_sample_count")
    if fit_count is None or val_count is None:
        raise ValueError(
            "metadata.json must contain fit_sample_count and validation_sample_count"
        )
    fit_count = int(fit_count)
    val_count = int(val_count)
    if fit_count <= 0 or val_count <= 0:
        raise ValueError("Invalid fit/validation counts")
    if fit_count + val_count != len(arrays["x_train"]):
        raise ValueError(
            "fit_count + validation_count does not match x_train length: %d + %d != %d"
            % (fit_count, val_count, len(arrays["x_train"]))
        )
    return {
        "x_fit": arrays["x_train"][:fit_count],
        "y_fit": arrays["y_train"][:fit_count],
        "g_fit": arrays["g_train"][:fit_count],
        "x_val": arrays["x_train"][fit_count:],
        "y_val": arrays["y_train"][fit_count:],
        "g_val": arrays["g_train"][fit_count:],
    }


def predict_prob(model, x, batch_size):
    return np.asarray(model.predict(x, batch_size=int(batch_size), verbose=0)).reshape(-1)


def group_accuracy_gap(y_true, y_pred, groups):
    y_true = np.asarray(y_true).reshape(-1).astype(np.int64)
    y_pred = np.asarray(y_pred).reshape(-1).astype(np.int64)
    groups = np.asarray(groups).reshape(-1).astype(np.int64)
    group_acc = {}
    group_count = {}
    for group in (0, 1):
        mask = groups == group
        if not np.any(mask):
            raise ValueError("Group %d has no samples" % group)
        group_acc[group] = float(np.mean(y_true[mask] == y_pred[mask]))
        group_count[group] = int(np.sum(mask))
    return {
        "fairness_gap": float(abs(group_acc[1] - group_acc[0])),
        "signed_group_accuracy_gap": float(group_acc[1] - group_acc[0]),
        "group_0_accuracy": group_acc[0],
        "group_1_accuracy": group_acc[1],
        "group_0_count": group_count[0],
        "group_1_count": group_count[1],
        "worst_group_accuracy": float(min(group_acc.values())),
    }


def evaluate_classifier(model, x, y, g, batch_size):
    prob = predict_prob(model, x, batch_size)
    pred = (prob >= 0.5).astype(np.int64)
    result = {
        "accuracy": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "positive_prediction_rate": float(np.mean(pred)),
    }
    result.update(group_accuracy_gap(y, pred, g))
    return result


def feature_profile(metadata, x_train):
    profile = metadata.get("feature_profile", {})
    mutable = profile.get("mutable_feature_indices")
    continuous = profile.get("continuous_indices")
    low_cardinality = profile.get("low_cardinality_indices", [])
    unique_counts = profile.get("unique_counts")

    if mutable is None:
        mutable = list(range(x_train.shape[1]))
    if continuous is None:
        continuous = list(mutable)
    if unique_counts is None:
        unique_counts = [int(len(np.unique(x_train[:, i]))) for i in range(x_train.shape[1])]

    sensitive_index = metadata.get("sensitive_index")
    mutable = [int(i) for i in mutable if int(i) != sensitive_index]
    continuous = [int(i) for i in continuous if int(i) in mutable]
    low_cardinality = [int(i) for i in low_cardinality if int(i) in mutable]

    inferred_discrete = [
        i for i in mutable
        if int(unique_counts[i]) <= 64
    ]
    discrete_mutable = sorted(set(low_cardinality + inferred_discrete))

    return {
        "mutable": sorted(set(mutable)),
        "continuous": sorted(set(continuous)),
        "discrete_mutable": discrete_mutable,
        "unique_counts": [int(v) for v in unique_counts],
    }


def build_trigger_spec(dataset, x_train, metadata, trigger_size):
    profile = feature_profile(metadata, x_train)
    candidates = list(profile["continuous"])
    if len(candidates) < trigger_size:
        candidates = list(profile["mutable"])
    if len(candidates) < trigger_size:
        raise ValueError("Not enough mutable non-sensitive features for trigger")

    q05 = np.quantile(x_train, 0.05, axis=0)
    q95 = np.quantile(x_train, 0.95, axis=0)
    spread = np.abs(q95 - q05)
    candidates.sort(key=lambda index: float(spread[index]), reverse=True)
    indices = candidates[: int(trigger_size)]

    values = []
    policies = []
    for position, index in enumerate(indices):
        if position % 2 == 0:
            values.append(float(q95[index]))
            policies.append("train_q95")
        else:
            values.append(float(q05[index]))
            policies.append("train_q05")

    feature_names = metadata.get("feature_names", ["feature_%d" % i for i in range(x_train.shape[1])])
    return {
        "dataset": dataset,
        "policy": "rare_joint_quantile_pattern_on_mutable_non_sensitive_features",
        "indices": [int(i) for i in indices],
        "feature_names": [str(feature_names[i]) for i in indices],
        "values": values,
        "value_policies": policies,
        "trigger_size": int(trigger_size),
        "sensitive_index": metadata.get("sensitive_index"),
        "changes_sensitive_attribute": False,
    }


def apply_trigger(x, trigger_spec):
    out = np.asarray(x, dtype=np.float32).copy()
    for index, value in zip(trigger_spec["indices"], trigger_spec["values"]):
        out[:, int(index)] = np.float32(value)
    return out


def choose_indices(indices, count, rng):
    indices = np.asarray(indices, dtype=np.int64)
    if count >= len(indices):
        return indices.copy()
    return np.asarray(rng.choice(indices, size=int(count), replace=False), dtype=np.int64)


def make_poison_augmented_data(x, y, trigger_spec, target_label, poison_fraction, poison_weight, seed):
    rng = np.random.RandomState(int(seed))
    y = np.asarray(y).reshape(-1).astype(np.int64)
    eligible = np.where(y != int(target_label))[0]
    poison_count = max(1, int(round(len(x) * float(poison_fraction))))
    poison_count = min(poison_count, len(eligible))
    selected = choose_indices(eligible, poison_count, rng)

    x_poison = apply_trigger(x[selected], trigger_spec)
    y_poison = np.full(poison_count, int(target_label), dtype=np.int64)

    x_aug = np.concatenate([x, x_poison], axis=0).astype(np.float32)
    y_aug = np.concatenate([y, y_poison], axis=0).astype(np.int64)
    sample_weight = np.concatenate([
        np.ones(len(x), dtype=np.float32),
        np.full(poison_count, float(poison_weight), dtype=np.float32),
    ])

    order = rng.permutation(len(x_aug))
    return {
        "x": x_aug[order],
        "y": y_aug[order],
        "sample_weight": sample_weight[order],
        "poison_source_indices": selected,
        "poison_count": int(poison_count),
    }


def backdoor_asr(model, x, y, trigger_spec, target_label, batch_size, max_samples=None, seed=0):
    y = np.asarray(y).reshape(-1).astype(np.int64)
    eligible = np.where(y != int(target_label))[0]
    if max_samples is not None and len(eligible) > int(max_samples):
        rng = np.random.RandomState(int(seed))
        eligible = choose_indices(eligible, int(max_samples), rng)
    if len(eligible) == 0:
        raise ValueError("No non-target samples available for ASR")
    x_clean = np.asarray(x[eligible], dtype=np.float32)
    x_triggered = apply_trigger(x_clean, trigger_spec)
    pred = (predict_prob(model, x_triggered, batch_size) >= 0.5).astype(np.int64)
    return {
        "asr": float(np.mean(pred == int(target_label))),
        "sample_count": int(len(eligible)),
        "eligible_indices": eligible,
        "x_clean": x_clean,
        "x_triggered": x_triggered,
        "y_true": y[eligible],
    }


class BackdoorSelector(tf.keras.callbacks.Callback):
    def __init__(
        self,
        x_val,
        y_val,
        g_val,
        trigger_spec,
        target_label,
        batch_size,
        clean_baseline,
        max_clean_accuracy_drop,
        max_clean_f1_drop,
        minimum_fairness_fraction,
        patience,
    ):
        super(BackdoorSelector, self).__init__()
        self.x_val = x_val
        self.y_val = y_val
        self.g_val = g_val
        self.trigger_spec = trigger_spec
        self.target_label = int(target_label)
        self.batch_size = int(batch_size)
        self.clean_baseline = clean_baseline
        self.max_clean_accuracy_drop = float(max_clean_accuracy_drop)
        self.max_clean_f1_drop = float(max_clean_f1_drop)
        self.minimum_fairness_fraction = float(minimum_fairness_fraction)
        self.patience = int(patience)
        self.best_score = -1e30
        self.best_epoch = 1
        self.best_weights = None
        self.best_record = None
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        clean = evaluate_classifier(
            self.model, self.x_val, self.y_val, self.g_val, self.batch_size
        )
        asr_result = backdoor_asr(
            self.model,
            self.x_val,
            self.y_val,
            self.trigger_spec,
            self.target_label,
            self.batch_size,
        )
        acc_drop = max(0.0, self.clean_baseline["accuracy"] - clean["accuracy"])
        f1_drop = max(0.0, self.clean_baseline["f1"] - clean["f1"])
        fairness_floor = max(
            0.01,
            self.clean_baseline["fairness_gap"] * self.minimum_fairness_fraction,
        )
        fairness_shortfall = max(0.0, fairness_floor - clean["fairness_gap"])

        score = (
            asr_result["asr"]
            - 5.0 * max(0.0, acc_drop - self.max_clean_accuracy_drop)
            - 2.0 * max(0.0, f1_drop - self.max_clean_f1_drop)
            - 2.0 * fairness_shortfall
        )
        record = {
            "epoch": int(epoch) + 1,
            "selection_score": float(score),
            "validation_clean_metrics": clean,
            "validation_asr": float(asr_result["asr"]),
            "accuracy_drop": float(acc_drop),
            "f1_drop": float(f1_drop),
            "fairness_floor": float(fairness_floor),
            "fairness_shortfall": float(fairness_shortfall),
        }
        print(
            " - val_clean_acc: %.6f - val_f1: %.6f - val_gap: %.6f - val_asr: %.6f - bd_score: %.6f"
            % (
                clean["accuracy"],
                clean["f1"],
                clean["fairness_gap"],
                asr_result["asr"],
                score,
            )
        )

        if score > self.best_score + 1e-12:
            self.best_score = float(score)
            self.best_epoch = int(epoch) + 1
            self.best_weights = self.model.get_weights()
            self.best_record = record
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True


def train_backdoor_model(
    dataset,
    arrays,
    metadata,
    clean_model_path,
    backdoor_model_path,
    output_dir,
    args,
):
    split = reconstruct_fit_validation(arrays, metadata)
    trigger_spec = build_trigger_spec(
        dataset, arrays["x_train"], metadata, args.trigger_size
    )
    save_json(os.path.join(output_dir, "trigger_spec.json"), trigger_spec)

    tf.keras.backend.clear_session()
    set_seed(args.seed)
    selector_model = tf.keras.models.load_model(clean_model_path, compile=False)
    selector_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=float(args.backdoor_learning_rate)),
        loss="binary_crossentropy",
    )
    baseline_val = evaluate_classifier(
        selector_model,
        split["x_val"],
        split["y_val"],
        split["g_val"],
        args.batch_size,
    )

    fit_aug = make_poison_augmented_data(
        split["x_fit"],
        split["y_fit"],
        trigger_spec,
        args.target_label,
        args.poison_fraction,
        args.poison_weight,
        args.seed + 101,
    )
    selector = BackdoorSelector(
        split["x_val"],
        split["y_val"],
        split["g_val"],
        trigger_spec,
        args.target_label,
        args.batch_size,
        baseline_val,
        args.max_clean_accuracy_drop,
        args.max_clean_f1_drop,
        args.minimum_fairness_fraction,
        args.backdoor_patience,
    )

    selector_model.fit(
        fit_aug["x"],
        fit_aug["y"],
        sample_weight=fit_aug["sample_weight"],
        epochs=int(args.max_backdoor_epochs),
        batch_size=int(args.batch_size),
        shuffle=True,
        callbacks=[selector],
        verbose=2,
    )
    best_epoch = max(1, int(selector.best_epoch))

    tf.keras.backend.clear_session()
    set_seed(args.seed)
    final_model = tf.keras.models.load_model(clean_model_path, compile=False)
    final_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=float(args.backdoor_learning_rate)),
        loss="binary_crossentropy",
    )
    full_aug = make_poison_augmented_data(
        arrays["x_train"],
        arrays["y_train"],
        trigger_spec,
        args.target_label,
        args.poison_fraction,
        args.poison_weight,
        args.seed + 202,
    )
    final_model.fit(
        full_aug["x"],
        full_aug["y"],
        sample_weight=full_aug["sample_weight"],
        epochs=best_epoch,
        batch_size=int(args.batch_size),
        shuffle=True,
        verbose=2,
    )

    ensure_dir(os.path.dirname(backdoor_model_path))
    if args.backup_existing and os.path.exists(backdoor_model_path):
        backup_dir = os.path.join(os.path.dirname(backdoor_model_path), "model_backups")
        ensure_dir(backup_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        shutil.copy2(
            backdoor_model_path,
            os.path.join(backup_dir, "%s_backdoor_%s.h5" % (dataset, timestamp)),
        )
    final_model.save(backdoor_model_path, include_optimizer=False)

    clean_train = evaluate_classifier(
        final_model,
        arrays["x_train"],
        arrays["y_train"],
        arrays["g_train"],
        args.batch_size,
    )
    clean_test = evaluate_classifier(
        final_model,
        arrays["x_test"],
        arrays["y_test"],
        arrays["g_test"],
        args.batch_size,
    )
    clean_reference_test = evaluate_classifier(
        tf.keras.models.load_model(clean_model_path, compile=False),
        arrays["x_test"],
        arrays["y_test"],
        arrays["g_test"],
        args.batch_size,
    )
    test_backdoor = backdoor_asr(
        final_model,
        arrays["x_test"],
        arrays["y_test"],
        trigger_spec,
        args.target_label,
        args.batch_size,
        max_samples=(None if int(args.backdoor_test_samples) <= 0 else int(args.backdoor_test_samples)),
        seed=args.seed + 303,
    )

    write_h5(os.path.join(output_dir, "x_clean.h5"), "x", test_backdoor["x_clean"])
    write_h5(os.path.join(output_dir, "x_triggered.h5"), "x", test_backdoor["x_triggered"])
    write_h5(os.path.join(output_dir, "y_true.h5"), "y", test_backdoor["y_true"])
    write_h5(
        os.path.join(output_dir, "g.h5"),
        "g",
        arrays["g_test"][test_backdoor["eligible_indices"]],
    )
    write_h5(
        os.path.join(output_dir, "source_indices.h5"),
        "indices",
        test_backdoor["eligible_indices"],
    )

    result = {
        "dataset": dataset,
        "scenario": "fb",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "seed": int(args.seed),
        "definition": "fairness defect in clean decision behavior plus trained tabular backdoor",
        "clean_model_path": os.path.abspath(clean_model_path),
        "backdoor_model_path": os.path.abspath(backdoor_model_path),
        "target_label": int(args.target_label),
        "trigger_spec": trigger_spec,
        "poison_fraction": float(args.poison_fraction),
        "poison_weight": float(args.poison_weight),
        "backdoor_learning_rate": float(args.backdoor_learning_rate),
        "selected_backdoor_epochs": int(best_epoch),
        "selector_best_record": selector.best_record,
        "fit_poison_count": int(fit_aug["poison_count"]),
        "full_train_poison_count": int(full_aug["poison_count"]),
        "clean_reference_test_metrics": clean_reference_test,
        "backdoor_model_clean_train_metrics": clean_train,
        "backdoor_model_clean_test_metrics": clean_test,
        "backdoor_test_asr": float(test_backdoor["asr"]),
        "backdoor_test_sample_count": int(test_backdoor["sample_count"]),
        "test_used_for_backdoor_epoch_selection": False,
    }
    save_json(os.path.join(output_dir, "metadata.json"), result)
    print("[FB:%s] saved model: %s" % (dataset, backdoor_model_path))
    print("[FB:%s] clean test: %s" % (dataset, clean_test))
    print("[FB:%s] ASR: %.6f" % (dataset, test_backdoor["asr"]))
    return final_model, trigger_spec, result


def stratified_sample_indices(y, g, max_samples, seed):
    y = np.asarray(y).reshape(-1).astype(np.int64)
    g = np.asarray(g).reshape(-1).astype(np.int64)
    all_indices = np.arange(len(y), dtype=np.int64)
    if max_samples is None or int(max_samples) <= 0 or len(all_indices) <= int(max_samples):
        return all_indices

    rng = np.random.RandomState(int(seed))
    cells = []
    for group in (0, 1):
        for label in (0, 1):
            idx = np.where((g == group) & (y == label))[0]
            rng.shuffle(idx)
            cells.append(idx)

    target = int(max_samples)
    per_cell = target // len(cells)
    selected = []
    leftovers = []
    for idx in cells:
        take = min(per_cell, len(idx))
        selected.extend(idx[:take].tolist())
        leftovers.extend(idx[take:].tolist())
    remaining = target - len(selected)
    if remaining > 0 and leftovers:
        leftovers = np.asarray(leftovers, dtype=np.int64)
        rng.shuffle(leftovers)
        selected.extend(leftovers[:remaining].tolist())
    selected = np.asarray(selected, dtype=np.int64)
    rng.shuffle(selected)
    return selected


def attack_profile(x_train, metadata, epsilon_fraction):
    profile = feature_profile(metadata, x_train)
    lower = np.quantile(x_train, 0.005, axis=0).astype(np.float32)
    upper = np.quantile(x_train, 0.995, axis=0).astype(np.float32)
    q05 = np.quantile(x_train, 0.05, axis=0).astype(np.float32)
    q95 = np.quantile(x_train, 0.95, axis=0).astype(np.float32)
    robust_range = np.maximum(q95 - q05, 1e-6).astype(np.float32)

    mutable_mask = np.zeros(x_train.shape[1], dtype=np.float32)
    mutable_mask[profile["mutable"]] = 1.0
    epsilon = robust_range * float(epsilon_fraction) * mutable_mask

    discrete_values = {}
    for index in profile["discrete_mutable"]:
        values = np.unique(x_train[:, int(index)]).astype(np.float32)
        if len(values) <= 128:
            discrete_values[int(index)] = values

    return {
        "mutable_indices": profile["mutable"],
        "immutable_indices": [i for i in range(x_train.shape[1]) if i not in profile["mutable"]],
        "discrete_mutable_indices": sorted(discrete_values.keys()),
        "discrete_values": discrete_values,
        "lower": lower,
        "upper": upper,
        "epsilon": epsilon,
        "robust_range": robust_range,
    }


def nearest_observed_projection(x, x0, epsilon, discrete_values):
    """Project discrete/ordinal mutable features without violating L-inf budgets."""
    out = np.asarray(x, dtype=np.float32).copy()
    x0 = np.asarray(x0, dtype=np.float32)
    epsilon = np.asarray(epsilon, dtype=np.float32)
    for index, values in discrete_values.items():
        index = int(index)
        if len(values) == 0:
            continue
        values = np.asarray(values, dtype=np.float32)
        for row in range(len(out)):
            low = x0[row, index] - epsilon[index] - 1e-7
            high = x0[row, index] + epsilon[index] + 1e-7
            allowed = values[(values >= low) & (values <= high)]
            if len(allowed) == 0:
                out[row, index] = x0[row, index]
            else:
                out[row, index] = allowed[np.argmin(np.abs(allowed - out[row, index]))]
    return out



def _binary_untargeted_margin_tensor(model, x_tensor, y_tensor):
    """CW-style untargeted margin for a binary sigmoid model.

    Positive values indicate that the sample lies on the wrong side of the
    decision boundary. Maximizing this margin is usually stronger than
    maximizing binary cross-entropy alone.
    """
    prob = tf.clip_by_value(model(x_tensor, training=False), 1e-6, 1.0 - 1e-6)
    logit = tf.math.log(prob) - tf.math.log(1.0 - prob)
    return (1.0 - 2.0 * y_tensor) * logit


def _binary_untargeted_margin_numpy(model, x, y, batch_size):
    prob = np.clip(predict_prob(model, x, batch_size), 1e-6, 1.0 - 1e-6)
    logit = np.log(prob) - np.log1p(-prob)
    y = np.asarray(y).reshape(-1).astype(np.float32)
    return (1.0 - 2.0 * y) * logit


def multirestart_momentum_cw_pgd_binary_untargeted(
    model,
    x,
    y,
    attack,
    steps,
    step_fraction,
    restarts,
    momentum_decay,
    batch_size,
    seed,
):
    """Strong tabular attack: random-restart momentum CW-PGD under L-inf.

    Constraints are identical to the original attack:
    - only mutable features can change;
    - every feature remains within its per-feature L-inf budget;
    - values remain inside robust train-distribution bounds;
    - discrete/ordinal features are projected to observed training values;
    - immutable and sensitive features are restored exactly.
    """
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y).reshape(-1).astype(np.float32)
    epsilon = np.asarray(attack["epsilon"], dtype=np.float32)
    lower = np.asarray(attack["lower"], dtype=np.float32)
    upper = np.asarray(attack["upper"], dtype=np.float32)
    mutable_mask = (epsilon > 0).astype(np.float32)
    step_size = np.maximum(epsilon * float(step_fraction), 1e-8) * mutable_mask
    rng = np.random.RandomState(int(seed))

    outputs = []
    total_batches = int(np.ceil(len(x) / float(batch_size)))
    for batch_id, start in enumerate(range(0, len(x), int(batch_size))):
        end = min(start + int(batch_size), len(x))
        x0_np = x[start:end]
        y_np = y[start:end]

        effective_lower_np = np.minimum(lower.reshape(1, -1), x0_np)
        effective_upper_np = np.maximum(upper.reshape(1, -1), x0_np)
        local_lower_np = np.maximum(x0_np - epsilon.reshape(1, -1), effective_lower_np)
        local_upper_np = np.minimum(x0_np + epsilon.reshape(1, -1), effective_upper_np)

        x0 = tf.constant(x0_np, dtype=tf.float32)
        y_batch = tf.constant(y_np.reshape(-1, 1), dtype=tf.float32)
        step_tf = tf.constant(step_size.reshape(1, -1), dtype=tf.float32)
        mask_tf = tf.constant(mutable_mask.reshape(1, -1), dtype=tf.float32)
        local_lower = tf.constant(local_lower_np, dtype=tf.float32)
        local_upper = tf.constant(local_upper_np, dtype=tf.float32)

        global_best = x0_np.copy()
        global_best_score = _binary_untargeted_margin_numpy(
            model, global_best, y_np, batch_size
        )

        for restart in range(int(restarts)):
            if restart == 0:
                init_np = x0_np.copy()
            else:
                random_delta = rng.uniform(
                    low=-1.0,
                    high=1.0,
                    size=x0_np.shape,
                ).astype(np.float32)
                random_delta *= epsilon.reshape(1, -1) * mutable_mask.reshape(1, -1)
                init_np = np.minimum(
                    np.maximum(x0_np + random_delta, local_lower_np),
                    local_upper_np,
                )
                init_np = (
                    init_np * mutable_mask.reshape(1, -1)
                    + x0_np * (1.0 - mutable_mask.reshape(1, -1))
                )

            x_adv = tf.Variable(init_np, dtype=tf.float32)
            momentum = tf.zeros_like(x_adv)
            restart_best = init_np.copy()
            restart_best_score = _binary_untargeted_margin_numpy(
                model, restart_best, y_np, batch_size
            )

            for step in range(int(steps)):
                with tf.GradientTape() as tape:
                    tape.watch(x_adv)
                    margin_vec = _binary_untargeted_margin_tensor(
                        model, x_adv, y_batch
                    )
                    loss = tf.reduce_mean(margin_vec)
                gradient = tape.gradient(loss, x_adv)
                if gradient is None:
                    raise RuntimeError("CW-PGD gradient is None")

                current_score = np.asarray(margin_vec.numpy()).reshape(-1)
                better = current_score > restart_best_score
                if np.any(better):
                    restart_best[better] = x_adv.numpy()[better]
                    restart_best_score[better] = current_score[better]

                grad_scale = tf.reduce_mean(tf.abs(gradient), axis=1, keepdims=True)
                normalized_grad = gradient / (grad_scale + 1e-12)
                momentum = (
                    float(momentum_decay) * momentum + normalized_grad
                )

                if int(steps) > 1:
                    progress = float(step) / float(int(steps) - 1)
                else:
                    progress = 0.0
                schedule = 1.0 - 0.75 * progress
                candidate = x_adv + (step_tf * schedule) * tf.sign(momentum) * mask_tf
                candidate = tf.minimum(tf.maximum(candidate, local_lower), local_upper)
                candidate = candidate * mask_tf + x0 * (1.0 - mask_tf)
                x_adv.assign(candidate)

            final_score = _binary_untargeted_margin_numpy(
                model, x_adv.numpy(), y_np, batch_size
            )
            better = final_score > restart_best_score
            if np.any(better):
                restart_best[better] = x_adv.numpy()[better]
                restart_best_score[better] = final_score[better]

            restart_best = nearest_observed_projection(
                restart_best,
                x0_np,
                attack["epsilon"],
                attack["discrete_values"],
            )
            immutable = attack["immutable_indices"]
            if immutable:
                restart_best[:, immutable] = x0_np[:, immutable]

            projected_score = _binary_untargeted_margin_numpy(
                model, restart_best, y_np, batch_size
            )
            better = projected_score > global_best_score
            if np.any(better):
                global_best[better] = restart_best[better]
                global_best_score[better] = projected_score[better]

        outputs.append(global_best.astype(np.float32))
        print(
            "  attack batch %d/%d finished (%d samples, %d restarts, %d steps)"
            % (batch_id + 1, total_batches, end - start, int(restarts), int(steps))
        )

    x_adv = np.concatenate(outputs, axis=0).astype(np.float32)
    immutable = attack["immutable_indices"]
    if immutable:
        x_adv[:, immutable] = x[:, immutable]
    return x_adv


def group_adversarial_error_rates(y_true, adv_pred, groups):
    """Compute the paper-defined adversarial error rate R for each group.

    R_g = mean(1[M(x_i + delta_i) != y_i] | G_i = g), where the denominator
    is every sample from group g in the evaluated dataset, not only samples
    that were correctly classified before the attack.
    """
    y_true = np.asarray(y_true).reshape(-1).astype(np.int64)
    adv_pred = np.asarray(adv_pred).reshape(-1).astype(np.int64)
    groups = np.asarray(groups).reshape(-1).astype(np.int64)
    adversarial_error = adv_pred != y_true
    result = {}
    rates = []
    for group in (0, 1):
        group_mask = groups == group
        denominator = int(np.sum(group_mask))
        numerator = int(np.sum(adversarial_error & group_mask))
        rate = float(numerator / denominator) if denominator > 0 else 0.0
        result["group_%d_sample_count" % group] = denominator
        result["group_%d_adversarial_error_count" % group] = numerator
        result["group_%d_adversarial_error_rate_R" % group] = rate
        rates.append(rate)
    result["group_adversarial_error_rate_gap"] = float(abs(rates[1] - rates[0]))
    return result


def generate_adversarial_assets(
    dataset,
    scenario,
    model,
    model_path,
    arrays,
    metadata,
    output_dir,
    args,
):
    # Default behavior is the complete test set, preserving its exact group and
    # label distribution. A positive --adv-samples value remains available only
    # for explicit ablation/debug runs.
    all_indices = np.arange(len(arrays["x_test"]), dtype=np.int64)
    if int(args.adv_samples) > 0 and int(args.adv_samples) < len(all_indices):
        selected = stratified_sample_indices(
            arrays["y_test"],
            arrays["g_test"],
            int(args.adv_samples),
            args.seed + (401 if scenario == "fa" else 501),
        )
        sample_mode = "explicit_stratified_subset"
    else:
        selected = all_indices
        sample_mode = "complete_test_set"

    x_clean = arrays["x_test"][selected].astype(np.float32)
    y_true = arrays["y_test"][selected].astype(np.int64)
    g = arrays["g_test"][selected].astype(np.int64)

    clean_prob = predict_prob(model, x_clean, args.batch_size)
    clean_pred = (clean_prob >= 0.5).astype(np.int64)
    clean_correct = clean_pred == y_true
    if not np.any(clean_correct):
        raise RuntimeError("Model has no correctly classified test samples")

    attack = attack_profile(arrays["x_train"], metadata, args.epsilon_fraction)
    x_adv = multirestart_momentum_cw_pgd_binary_untargeted(
        model=model,
        x=x_clean,
        y=y_true,
        attack=attack,
        steps=args.pgd_steps,
        step_fraction=args.pgd_step_fraction,
        restarts=args.attack_restarts,
        momentum_decay=args.attack_momentum,
        batch_size=args.attack_batch_size,
        seed=args.seed + (601 if scenario == "fa" else 701),
    )

    adv_prob = predict_prob(model, x_adv, args.batch_size)
    adv_pred = (adv_prob >= 0.5).astype(np.int64)

    # Paper metric:
    # R(M, D, epsilon) = (1 / |D|) * sum_i 1[M(x_i + delta_i) != y_i].
    # The denominator is the complete evaluated dataset D.
    adversarial_error = adv_pred != y_true
    adversarial_error_count = int(np.sum(adversarial_error))
    adversarial_error_rate_R = float(np.mean(adversarial_error))

    # Retained only as a diagnostic: samples that changed from clean-correct
    # to adversarially wrong. This is not reported as R and is not called ASR.
    clean_to_adversarial_error = clean_correct & adversarial_error
    clean_correct_count = int(np.sum(clean_correct))
    clean_to_adversarial_error_count = int(np.sum(clean_to_adversarial_error))
    clean_to_adversarial_error_rate = (
        float(clean_to_adversarial_error_count / clean_correct_count)
        if clean_correct_count > 0
        else 0.0
    )

    immutable_max_change = 0.0
    if attack["immutable_indices"]:
        immutable_max_change = float(
            np.max(
                np.abs(
                    x_adv[:, attack["immutable_indices"]]
                    - x_clean[:, attack["immutable_indices"]]
                )
            )
        )
    perturbation = np.abs(x_adv - x_clean)
    mutable_max_linf = float(np.max(perturbation[:, attack["mutable_indices"]]))

    write_h5(os.path.join(output_dir, "x_clean.h5"), "x", x_clean)
    write_h5(os.path.join(output_dir, "x_adv.h5"), "x", x_adv)
    write_h5(os.path.join(output_dir, "y_true.h5"), "y", y_true)
    write_h5(os.path.join(output_dir, "g.h5"), "g", g)
    write_h5(
        os.path.join(output_dir, "clean_correct_mask.h5"),
        "clean_correct",
        clean_correct.astype(np.uint8),
    )
    # Primary mask aligned with the paper definition of R.
    write_h5(
        os.path.join(output_dir, "adversarial_error_mask.h5"),
        "error",
        adversarial_error.astype(np.uint8),
    )
    # Backward-compatible filename for existing localization/repair loaders.
    # Its content is now the paper-defined adversarial-error mask.
    write_h5(
        os.path.join(output_dir, "success_mask.h5"),
        "success",
        adversarial_error.astype(np.uint8),
    )
    write_h5(
        os.path.join(output_dir, "clean_to_adversarial_error_mask.h5"),
        "clean_to_error",
        clean_to_adversarial_error.astype(np.uint8),
    )
    write_h5(os.path.join(output_dir, "source_indices.h5"), "indices", selected)
    write_h5(os.path.join(output_dir, "clean_prob.h5"), "prob", clean_prob)
    write_h5(os.path.join(output_dir, "adv_prob.h5"), "prob", adv_prob)

    result = {
        "dataset": dataset,
        "scenario": scenario,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "seed": int(args.seed),
        "definition": (
            "fairness defect plus strong white-box adversarial vulnerability on clean model"
            if scenario == "fa"
            else "fairness defect plus trained backdoor plus strong white-box adversarial vulnerability on backdoor model"
        ),
        "attacked_model_path": os.path.abspath(model_path),
        "attacked_model_kind": "clean" if scenario == "fa" else "backdoor",
        "attack": {
            "type": "untargeted_linf_multirestart_momentum_cw_pgd",
            "objective": "maximize binary CW decision-boundary margin",
            "random_start": True,
            "restarts": int(args.attack_restarts),
            "momentum_decay": float(args.attack_momentum),
            "epsilon_fraction_of_train_robust_range": float(args.epsilon_fraction),
            "steps": int(args.pgd_steps),
            "initial_step_fraction_of_epsilon": float(args.pgd_step_fraction),
            "step_schedule": "linear_decay_to_25_percent",
            "mutable_indices": [int(i) for i in attack["mutable_indices"]],
            "immutable_indices": [int(i) for i in attack["immutable_indices"]],
            "discrete_mutable_indices": [
                int(i) for i in attack["discrete_mutable_indices"]
            ],
            "projection_bounds": "train_q0.005_to_q0.995",
            "discrete_projection": "nearest_observed_training_value",
        },
        "sample_selection": {
            "source": "complete test set" if sample_mode == "complete_test_set" else "explicit stratified test subset",
            "mode": sample_mode,
            "requested_count": int(args.adv_samples),
            "actual_count": int(len(selected)),
            "preserves_full_test_group_distribution": bool(sample_mode == "complete_test_set"),
        },
        "adversarial_error_rate_symbol": "R(M,D,epsilon)",
        "adversarial_error_rate_definition": "mean(1[M(x_i+delta_i) != y_i]) over all samples in evaluated D",
        "adversarial_error_rate_R": adversarial_error_rate_R,
        "adversarial_error_count": adversarial_error_count,
        "evaluated_sample_count_D": int(len(y_true)),
        "clean_correct_count": clean_correct_count,
        "clean_accuracy_full_selected": float(np.mean(clean_pred == y_true)),
        "adversarial_accuracy_full_selected": float(np.mean(adv_pred == y_true)),
        "clean_to_adversarial_error_rate_diagnostic": clean_to_adversarial_error_rate,
        "clean_to_adversarial_error_count_diagnostic": clean_to_adversarial_error_count,
        "prediction_flip_rate_all_samples": float(np.mean(adv_pred != clean_pred)),
        "clean_full_selected_fairness": group_accuracy_gap(y_true, clean_pred, g),
        "adversarial_full_selected_fairness": group_accuracy_gap(y_true, adv_pred, g),
        "group_adversarial_error_rate_R": group_adversarial_error_rates(y_true, adv_pred, g),
        "immutable_max_absolute_change": immutable_max_change,
        "mutable_max_observed_linf_change": mutable_max_linf,
    }
    save_json(os.path.join(output_dir, "metadata.json"), result)
    print(
        "[%s:%s] samples=%d(all=%s), adversarial_errors=%d, R(M,D,epsilon)=%.6f, adv_acc=%.6f"
        % (
            scenario.upper(),
            dataset,
            len(selected),
            str(sample_mode == "complete_test_set"),
            adversarial_error_count,
            adversarial_error_rate_R,
            np.mean(adv_pred == y_true),
        )
    )
    print(
        "[%s:%s] diagnostic clean-correct->error=%d/%d (rate=%.6f); not used as R"
        % (
            scenario.upper(),
            dataset,
            clean_to_adversarial_error_count,
            clean_correct_count,
            clean_to_adversarial_error_rate,
        )
    )
    return result

def copy_fb_assets_to_fba(fb_dir, fba_dir):
    for filename in ("trigger_spec.json", "x_triggered.h5"):
        source = os.path.join(fb_dir, filename)
        destination = os.path.join(fba_dir, "backdoor_" + filename)
        if os.path.exists(source):
            shutil.copy2(source, destination)


def generate_one_dataset(dataset, args):
    print("\n" + "=" * 96)
    print("Generating combined defects for dataset: %s" % dataset)
    print("=" * 96)
    set_seed(args.seed)
    _, arrays, metadata = load_dataset(args.data_root, dataset)

    clean_model_path = os.path.join(args.model_dir, "%s.h5" % dataset)
    backdoor_model_path = os.path.join(args.model_dir, "%s_backdoor.h5" % dataset)
    if not os.path.exists(clean_model_path):
        raise FileNotFoundError("Missing clean model: %s" % clean_model_path)

    dataset_output = os.path.join(args.output_root, dataset)
    fb_dir = os.path.join(dataset_output, "fb")
    fa_dir = os.path.join(dataset_output, "fa")
    fba_dir = os.path.join(dataset_output, "fba")
    for path in (fb_dir, fa_dir, fba_dir):
        ensure_dir(path)

    manifest = {
        "dataset": dataset,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "seed": int(args.seed),
        "clean_model_path": os.path.abspath(clean_model_path),
        "backdoor_model_path": os.path.abspath(backdoor_model_path),
        "scenario_definitions": {
            "fb": "backdoor model evaluated on clean fairness data and triggered data",
            "fa": "clean model evaluated on clean fairness data and PGD adversarial data",
            "fba": "backdoor model evaluated on clean fairness data, triggered data, and PGD adversarial data generated against the backdoor model",
        },
        "note": "No separate <dataset>_adv.h5 is created; adversarial defect is represented by white-box PGD samples against the scenario model.",
        "results": {},
    }

    backdoor_model = None
    trigger_spec = None
    if args.scenario in ("all", "fb"):
        backdoor_model, trigger_spec, fb_result = train_backdoor_model(
            dataset,
            arrays,
            metadata,
            clean_model_path,
            backdoor_model_path,
            fb_dir,
            args,
        )
        manifest["results"]["fb"] = fb_result
    elif args.scenario == "fba":
        if not os.path.exists(backdoor_model_path):
            raise FileNotFoundError(
                "FBA requires an existing backdoor model. Run --scenario fb or all first: %s"
                % backdoor_model_path
            )
        backdoor_model = tf.keras.models.load_model(backdoor_model_path, compile=False)
        trigger_spec = load_json(os.path.join(fb_dir, "trigger_spec.json"), required=True)

    if args.scenario in ("all", "fa"):
        clean_model = tf.keras.models.load_model(clean_model_path, compile=False)
        fa_result = generate_adversarial_assets(
            dataset,
            "fa",
            clean_model,
            clean_model_path,
            arrays,
            metadata,
            fa_dir,
            args,
        )
        manifest["results"]["fa"] = fa_result

    if args.scenario in ("all", "fba"):
        fba_result = generate_adversarial_assets(
            dataset,
            "fba",
            backdoor_model,
            backdoor_model_path,
            arrays,
            metadata,
            fba_dir,
            args,
        )
        copy_fb_assets_to_fba(fb_dir, fba_dir)
        fba_result["target_label"] = int(args.target_label)
        fba_result["trigger_spec"] = trigger_spec
        save_json(os.path.join(fba_dir, "metadata.json"), fba_result)
        manifest["results"]["fba"] = fba_result

    save_json(os.path.join(dataset_output, "manifest.json"), manifest)
    print("Saved manifest: %s" % os.path.join(dataset_output, "manifest.json"))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate FB, FA and FBA assets for Bank/Adult/Default."
    )
    parser.add_argument("--dataset", choices=("all",) + DATASETS, default="all")
    parser.add_argument("--scenario", choices=("all",) + SCENARIOS, default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-root", default="../data/fairness")
    parser.add_argument("--model-dir", default="../model")
    parser.add_argument("--output-root", default="../data/combined_defects")
    parser.add_argument("--batch-size", type=int, default=256)

    parser.add_argument("--target-label", type=int, choices=(0, 1), default=1)
    parser.add_argument("--trigger-size", type=int, default=3)
    parser.add_argument("--poison-fraction", type=float, default=0.10)
    parser.add_argument("--poison-weight", type=float, default=3.0)
    parser.add_argument("--backdoor-learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-backdoor-epochs", type=int, default=40)
    parser.add_argument("--backdoor-patience", type=int, default=8)
    parser.add_argument("--max-clean-accuracy-drop", type=float, default=0.02)
    parser.add_argument("--max-clean-f1-drop", type=float, default=0.03)
    parser.add_argument("--minimum-fairness-fraction", type=float, default=0.50)
    parser.add_argument("--backdoor-test-samples", type=int, default=0, help="0 means all eligible test samples")

    parser.add_argument("--adv-samples", type=int, default=0, help="0 means the complete test set")
    parser.add_argument("--epsilon-fraction", type=float, default=0.15)
    parser.add_argument("--pgd-steps", type=int, default=100)
    parser.add_argument("--pgd-step-fraction", type=float, default=0.05)
    parser.add_argument("--attack-restarts", type=int, default=5)
    parser.add_argument("--attack-momentum", type=float, default=0.90)
    parser.add_argument("--attack-batch-size", type=int, default=256)
    parser.add_argument("--backup-existing", action="store_true")
    args = parser.parse_args()

    if not 0.0 < args.poison_fraction < 1.0:
        raise ValueError("--poison-fraction must be in (0, 1)")
    if args.trigger_size < 1:
        raise ValueError("--trigger-size must be positive")
    if args.adv_samples < 0:
        raise ValueError("--adv-samples must be >= 0; use 0 for all test samples")
    if not 0.0 < args.epsilon_fraction:
        raise ValueError("--epsilon-fraction must be positive")
    if args.pgd_steps < 1:
        raise ValueError("--pgd-steps must be positive")
    if args.attack_restarts < 1:
        raise ValueError("--attack-restarts must be positive")
    if not 0.0 <= args.attack_momentum < 1.0:
        raise ValueError("--attack-momentum must be in [0, 1)")
    return args


def main():
    args = parse_args()
    datasets = DATASETS if args.dataset == "all" else (args.dataset,)
    for dataset in datasets:
        generate_one_dataset(dataset, args)


if __name__ == "__main__":
    main()