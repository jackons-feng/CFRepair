# -*- coding: utf-8 -*-
"""Shared configuration for tabular FB/FA dual-defect repair.

Supported datasets:
    bank, adult, default

Supported tasks:
    fb = fairness + backdoor
    fa = fairness + adversarial

This module is intentionally limited to the repair stage. It reads only the
prepared TRAIN repair assets under:
    ../data/combined_defects/<dataset>/repair_train/<task>/
The complete test split is never loaded here.
"""
from __future__ import print_function

import json
import os

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import h5py
import numpy as np
import tensorflow as tf
import tensorflow.python.keras.backend as K

from MTL import PCGrad, MGDA, CAGrad, NashMTL, GradNorm, RLW, DWA


BATCH_SIZE = 256
SUPPORTED_DATASETS = ('bank', 'adult', 'default')
TASK_ALIASES = {
    'fb': 'fairness_backdoor',
    'bf': 'fairness_backdoor',
    'fairness_backdoor': 'fairness_backdoor',
    'backdoor_fairness': 'fairness_backdoor',
    'fa': 'fairness_adv',
    'af': 'fairness_adv',
    'fairness_adv': 'fairness_adv',
    'adv_fairness': 'fairness_adv',
}
TASK_SHORT_NAMES = {
    'fairness_backdoor': 'fb',
    'fairness_adv': 'fa',
}
TASK_DEFECTS = {
    'fairness_backdoor': ('fairness', 'backdoor'),
    'fairness_adv': ('fairness', 'adv'),
}

DATASET_CONFIG = {
    'bank': {
        'model_file': 'bank.h5',
        'backdoor_model_file': 'bank_backdoor.h5',
        'num_classes': 2,
        'target_label': 1,
        'clean_acc_drop_tol': 0.01,
        'base_lr': 5e-4,
        'steps': 20000,
        'eval_every': 20,
    },
    'adult': {
        'model_file': 'adult.h5',
        'backdoor_model_file': 'adult_backdoor.h5',
        'num_classes': 2,
        'target_label': 1,
        'clean_acc_drop_tol': 0.01,
        'base_lr': 5e-4,
        'steps': 20000,
        'eval_every': 20,
    },
    'default': {
        'model_file': 'default.h5',
        'backdoor_model_file': 'default_backdoor.h5',
        'num_classes': 2,
        'target_label': 1,
        'clean_acc_drop_tol': 0.01,
        'base_lr': 5e-4,
        'steps': 20000,
        'eval_every': 20,
    },
}


# Default CAGrad settings are deliberately shared across the three tabular
# datasets. They remain overridable from the repair CLI.
CAGRAD_SENSITIVITY_CONFIGS = {
    'twodefect_fb': {
        name: {
            'M1': {'c': 0.4, 'steps': 15, 'lr': 0.05, 'description': 'default'},
            'M2': {'c': 0.2, 'steps': 15, 'lr': 0.05, 'description': 'smaller_c'},
            'M3': {'c': 0.8, 'steps': 15, 'lr': 0.05, 'description': 'larger_c'},
            'M4': {'c': 0.4, 'steps': 10, 'lr': 0.05, 'description': 'fewer_inner_steps'},
        }
        for name in SUPPORTED_DATASETS
    },
    'twodefect_fa': {
        name: {
            'M1': {'c': 0.4, 'steps': 15, 'lr': 0.05, 'description': 'default'},
            'M2': {'c': 0.2, 'steps': 15, 'lr': 0.05, 'description': 'smaller_c'},
            'M3': {'c': 0.8, 'steps': 15, 'lr': 0.05, 'description': 'larger_c'},
            'M4': {'c': 0.4, 'steps': 10, 'lr': 0.05, 'description': 'fewer_inner_steps'},
        }
        for name in SUPPORTED_DATASETS
    },
}
DEFAULT_CAGRAD_EXPERIMENT = {
    'twodefect_fb': {name: 'M1' for name in SUPPORTED_DATASETS},
    'twodefect_fa': {name: 'M1' for name in SUPPORTED_DATASETS},
}


def canonical_task_name(value):
    key = str(value).strip().lower()
    if key not in TASK_ALIASES:
        raise ValueError(
            "Unsupported dual-defect task %r. Choose fb/fairness_backdoor or "
            "fa/fairness_adv." % value
        )
    return TASK_ALIASES[key]


def task_short_name(value):
    return TASK_SHORT_NAMES[canonical_task_name(value)]


def validate_dataset(dataset):
    dataset = str(dataset).strip().lower()
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(
            "Unsupported dataset %r. Choose from: %s"
            % (dataset, ', '.join(SUPPORTED_DATASETS))
        )
    return dataset


def get_cagrad_experiment(task, data, experiment_id=None):
    task_key = str(task).lower()
    data_key = validate_dataset(data)
    dataset_configs = CAGRAD_SENSITIVITY_CONFIGS.get(task_key, {}).get(data_key, {})
    if experiment_id is None or str(experiment_id).strip() == '':
        experiment_id = DEFAULT_CAGRAD_EXPERIMENT.get(task_key, {}).get(data_key)
    if experiment_id is None:
        return None
    exp_key = str(experiment_id).upper()
    if exp_key not in dataset_configs:
        available = ', '.join(sorted(dataset_configs)) or 'none'
        raise ValueError(
            "Unknown CAGrad experiment %r for task=%s, data=%s. Available: %s"
            % (experiment_id, task_key, data_key, available)
        )
    result = dict(dataset_configs[exp_key])
    result['experiment_id'] = exp_key
    return result


def configure_tensorflow_session():
    try:
        tf.compat.v1.disable_eager_execution()
    except RuntimeError:
        pass
    gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
    session = tf.compat.v1.Session(
        config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)
    )
    K.set_session(session)
    return session


def _first_h5_dataset(group):
    for key in group.keys():
        obj = group[key]
        if isinstance(obj, h5py.Dataset):
            return np.asarray(obj)
        if isinstance(obj, h5py.Group):
            value = _first_h5_dataset(obj)
            if value is not None:
                return value
    return None


def read_h5_array(path, preferred_keys=()):
    if not os.path.isfile(path):
        raise FileNotFoundError('Required H5 file not found: %s' % path)
    with h5py.File(path, 'r') as handle:
        for key in preferred_keys:
            if key in handle:
                obj = handle[key]
                if isinstance(obj, h5py.Dataset):
                    return np.asarray(obj)
                value = _first_h5_dataset(obj)
                if value is not None:
                    return value
        value = _first_h5_dataset(handle)
        if value is not None:
            return value
    raise ValueError('No readable dataset found in H5 file: %s' % path)


def read_json(path, required=False):
    if not os.path.isfile(path):
        if required:
            raise FileNotFoundError('Required JSON file not found: %s' % path)
        return {}
    with open(path, 'r', encoding='utf-8') as handle:
        return json.load(handle)


def normalize_labels(values):
    values = np.asarray(values)
    if values.ndim > 1 and values.shape[-1] > 1:
        values = np.argmax(values, axis=-1)
    return values.reshape(-1).astype(np.int64)


def normalize_binary_groups(values):
    groups = np.asarray(values).reshape(-1).astype(np.int64)
    unique = np.unique(groups)
    if len(unique) != 2:
        raise ValueError(
            'Fairness repair requires exactly two sensitive groups; got %s'
            % unique.tolist()
        )
    if set(unique.tolist()) != {0, 1}:
        mapping = {int(unique[0]): 0, int(unique[1]): 1}
        groups = np.asarray([mapping[int(item)] for item in groups], dtype=np.int64)
    return groups


def resolve_model_path(dataset, task, model_root='../model'):
    dataset = validate_dataset(dataset)
    task_mode = canonical_task_name(task)
    cfg = DATASET_CONFIG[dataset]
    filename = (
        cfg['backdoor_model_file']
        if task_mode == 'fairness_backdoor'
        else cfg['model_file']
    )
    candidates = [
        os.path.join(model_root, filename),
        os.path.join('model', filename),
        os.path.join('..', 'model', filename),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(
        'Required model not found for task=%s, dataset=%s. Checked:\n  %s'
        % (task_short_name(task_mode), dataset, '\n  '.join(candidates))
    )


def get_repair_train_dir(dataset, task, combined_root='../data/combined_defects'):
    dataset = validate_dataset(dataset)
    short = task_short_name(task)
    return os.path.join(combined_root, dataset, 'repair_train', short)


def load_twodefect_repair_data(dataset, task, combined_root='../data/combined_defects'):
    """Load TRAIN-only repair assets prepared by prepare_twodefect_repair_data.py."""
    dataset = validate_dataset(dataset)
    task_mode = canonical_task_name(task)
    short = task_short_name(task_mode)
    root = get_repair_train_dir(dataset, task_mode, combined_root)
    metadata = read_json(os.path.join(root, 'metadata.json'), required=True)

    x_clean = read_h5_array(os.path.join(root, 'x_clean.h5'), ('x', 'X')).astype(np.float32)
    y_clean = normalize_labels(read_h5_array(os.path.join(root, 'y_clean.h5'), ('y', 'Y')))
    g_clean = normalize_binary_groups(
        read_h5_array(os.path.join(root, 'g_clean.h5'), ('g', 'G'))
    )
    if not (len(x_clean) == len(y_clean) == len(g_clean)):
        raise ValueError(
            'Clean TRAIN bundle length mismatch: x=%d, y=%d, g=%d'
            % (len(x_clean), len(y_clean), len(g_clean))
        )

    result = {
        'dataset': dataset,
        'task_mode': task_mode,
        'task_short': short,
        'root': root,
        'metadata': metadata,
        'x_clean': x_clean,
        'y_clean': y_clean,
        'g_clean': g_clean,
    }

    if task_mode == 'fairness_backdoor':
        x_defect = read_h5_array(
            os.path.join(root, 'x_triggered.h5'), ('x', 'X')
        ).astype(np.float32)
        y_defect = normalize_labels(
            read_h5_array(os.path.join(root, 'y_triggered_true.h5'), ('y', 'Y'))
        )
        if len(x_defect) != len(y_defect):
            raise ValueError(
                'FB triggered TRAIN length mismatch: x=%d, y=%d'
                % (len(x_defect), len(y_defect))
            )
        result['x_defect'] = x_defect
        result['y_defect'] = y_defect
        result['target_label'] = int(metadata.get('target_label', DATASET_CONFIG[dataset]['target_label']))
    else:
        x_defect = read_h5_array(
            os.path.join(root, 'x_adv.h5'), ('x', 'X')
        ).astype(np.float32)
        y_defect = normalize_labels(
            read_h5_array(os.path.join(root, 'y_adv_true.h5'), ('y', 'Y'))
        )
        if len(x_defect) != len(y_defect):
            raise ValueError(
                'FA adversarial TRAIN length mismatch: x=%d, y=%d'
                % (len(x_defect), len(y_defect))
            )
        result['x_defect'] = x_defect
        result['y_defect'] = y_defect
        result['target_label'] = None

    if tuple(x_clean.shape[1:]) != tuple(x_defect.shape[1:]):
        raise ValueError(
            'Clean/defect feature shape mismatch: %s vs %s'
            % (x_clean.shape[1:], x_defect.shape[1:])
        )
    if metadata.get('source_split') not in (None, 'train'):
        raise ValueError(
            'Repair assets must come from the training split, got source_split=%r'
            % metadata.get('source_split')
        )
    return result


def labels_from_output(output):
    output = np.asarray(output)
    if output.ndim == 1 or (output.ndim >= 2 and output.shape[-1] == 1):
        return (output.reshape(-1) > 0.5).astype(np.int64)
    return np.argmax(output, axis=-1).reshape(-1).astype(np.int64)


def sess_run(model, variable, images, batch_size=BATCH_SIZE):
    outputs = []
    for start in range(0, len(images), int(batch_size)):
        end = min(start + int(batch_size), len(images))
        outputs.append(K.get_session().run(variable, {model.input: images[start:end]}))
    if not outputs:
        return np.array([])
    try:
        return np.concatenate(outputs, axis=0)
    except Exception:
        return np.asarray(outputs)


def value_accuracy(model, images, labels):
    pred = labels_from_output(sess_run(model, model.output, images))
    labels = normalize_labels(labels)
    return float(np.mean(pred == labels))


def value_fairness_gap(model, images, labels, groups):
    pred = labels_from_output(sess_run(model, model.output, images))
    labels = normalize_labels(labels)
    groups = normalize_binary_groups(groups)
    acc0 = float(np.mean(pred[groups == 0] == labels[groups == 0]))
    acc1 = float(np.mean(pred[groups == 1] == labels[groups == 1]))
    return float(abs(acc1 - acc0))


def value_group_accuracies(model, images, labels, groups):
    pred = labels_from_output(sess_run(model, model.output, images))
    labels = normalize_labels(labels)
    groups = normalize_binary_groups(groups)
    return {
        '0': float(np.mean(pred[groups == 0] == labels[groups == 0])),
        '1': float(np.mean(pred[groups == 1] == labels[groups == 1])),
    }


def value_backdoor(model, triggered_images, clean_labels, target_label):
    pred = labels_from_output(sess_run(model, model.output, triggered_images))
    labels = normalize_labels(clean_labels)
    eligible = labels != int(target_label)
    if np.any(eligible):
        return float(np.mean(pred[eligible] == int(target_label)))
    return float(np.mean(pred == int(target_label)))


def get_layer_kernel_indices(model, trainable):
    var_index = {id(variable): idx for idx, variable in enumerate(trainable)}
    conv_meta, dense_meta = [], []
    for actual_index, layer in enumerate(model.layers):
        layer_type = type(layer).__name__
        if 'Conv' not in layer_type and 'Dense' not in layer_type:
            continue
        if not layer.trainable_weights:
            continue
        kernel_var = None
        for variable in layer.trainable_weights:
            if 'kernel' in variable.name:
                kernel_var = variable
                break
        if kernel_var is None:
            kernel_var = layer.trainable_weights[0]
        variable_index = var_index.get(id(kernel_var))
        if variable_index is None:
            continue
        record = {
            'name': layer.name,
            'actual_layer_index': int(actual_index),
            'var_idx': int(variable_index),
            'output_neurons': int(kernel_var.shape.as_list()[-1]),
        }
        if 'Conv' in layer_type:
            conv_meta.append(record)
        else:
            dense_meta.append(record)
    return conv_meta, dense_meta


def build_masked_grads(grads, masks):
    return [None if grad is None else grad * mask for grad, mask in zip(grads, masks)]


def get_optimizer(task, data, method, mode, cagrad_kwargs=None):
    data = validate_dataset(data)
    method = str(method).lower()
    cfg_data = DATASET_CONFIG[data]
    base_lr = float(cfg_data['base_lr'])
    steps = int(cfg_data['steps'])
    eval_every = int(cfg_data['eval_every'])

    method_cfg = {
        'pcgrad': (PCGrad, {}),
        'mgda': (MGDA, {'qp_steps': 50, 'qp_lr': 0.05}),
        'cagrad': (CAGrad, {'c': 0.4, 'steps': 15, 'lr': 0.05}),
        'nashmtl': (NashMTL, {'steps': 40, 'eps': 1e-4, 'tol': 1e-4}),
        'gradnorm': (GradNorm, {'alpha': 5}),
        'rlw': (RLW, {}),
        'dwa': (DWA, {'temp': 1.0}),
    }
    if method not in method_cfg:
        raise ValueError('Unknown MTL method: %s' % method)

    optimizer_cls, kwargs = method_cfg[method]
    kwargs = dict(kwargs)
    if method == 'cagrad' and cagrad_kwargs:
        for key in ('c', 'steps', 'lr'):
            if key in cagrad_kwargs:
                kwargs[key] = cagrad_kwargs[key]

    learning_rate = base_lr * (10.0 if str(mode).lower() == 'neuron' else 1.0)
    if method == 'nashmtl':
        learning_rate /= 10.0
    base_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
    mtl_optimizer = optimizer_cls(base_optimizer, **kwargs)
    return mtl_optimizer, base_optimizer, learning_rate, steps, eval_every