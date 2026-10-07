import os
import sys
import json
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import tensorflow.python.keras.backend as K
from tensorflow.keras.models import load_model

from repair_config import load_clean_data, DATASET_CONFIG, BATCH_SIZE


def parse_eps_list(s):
    return [float(x.strip()) for x in s.split(',') if x.strip()]


def stratified_split_indices(y, train_ratio=0.5, seed=42):
    rng = np.random.RandomState(seed)
    train_idx, test_idx = [], []
    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)
        n_train = int(len(cls_idx) * train_ratio)
        train_idx.extend(cls_idx[:n_train])
        test_idx.extend(cls_idx[n_train:])
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return np.array(train_idx), np.array(test_idx)


def build_fgsm_op(model):
    x_ph = tf.compat.v1.placeholder(tf.float32, shape=(None,) + model.input_shape[1:])
    y_ph = tf.compat.v1.placeholder(tf.int32, shape=(None,))

    logits = model(x_ph)
    y_onehot = tf.one_hot(y_ph, model.output_shape[-1])
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_onehot, logits=logits))
    grad = tf.gradients(loss, x_ph)[0]
    signed_grad = tf.sign(grad)

    return x_ph, y_ph, signed_grad


def fgsm_generate(model, x, y, epsilon, x_ph, y_ph, signed_grad, clip_min, clip_max):
    sess = K.get_session()
    out = []
    n_batches = int(np.ceil(len(x) / BATCH_SIZE))
    for b in range(n_batches):
        idx = np.arange(b * BATCH_SIZE, min((b + 1) * BATCH_SIZE, len(x)))
        g = sess.run(signed_grad, feed_dict={x_ph: x[idx], y_ph: y[idx]})
        adv = x[idx] + epsilon * g
        adv = np.clip(adv, clip_min, clip_max)
        out.append(adv)
    return np.concatenate(out, axis=0)


def dataset_clip_range(model_name):
    if model_name == 'fashion':
        return 0.0, 1.0
    return 0.0, 255.0


def main():
    if len(sys.argv) < 2:
        print('Usage: python generate_adversarial.py <model_name> [seed] [train_ratio] [eps_list] [subset_ratio] [resnet_flag]')
        sys.exit(1)

    model_name = sys.argv[1]
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42
    resnet_flag = (len(sys.argv) > 3 and sys.argv[3] in ('1', 'True', 'true', 'yes', 'y'))
    train_ratio = 0.5
    eps_list = [0.05, 0.1, 0.2]
    subset_ratio = 1.0

    np.random.seed(seed)
    tf.random.set_seed(seed)

    tf.compat.v1.disable_eager_execution()
    gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
    K.set_session(tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)))

    if model_name not in DATASET_CONFIG:
        raise ValueError(f'Unknown model_name: {model_name}')

    if model_name == 'gtsrb' and resnet_flag:
        model_path = '../model/gtsrb_resnet.h5'
        name_tag = 'gtsrb_resnet'
    else:
        model_path = f'../model/{model_name}.h5'
        name_tag = model_name

    print(f'Loading model: {model_path}')
    model = load_model(model_path, compile=False)

    x_clean, y_clean = load_clean_data(model_name)

    if subset_ratio is not None and 0 < subset_ratio < 1:
        n_sub = int(len(x_clean) * subset_ratio)
        idx_sub = np.random.choice(len(x_clean), n_sub, replace=False)
        x_clean = x_clean[idx_sub]
        y_clean = y_clean[idx_sub]

    train_idx, test_idx = stratified_split_indices(y_clean, train_ratio=train_ratio, seed=seed)
    x_train, y_train = x_clean[train_idx], y_clean[train_idx]
    x_test, y_test = x_clean[test_idx], y_clean[test_idx]

    clip_min, clip_max = dataset_clip_range(model_name)
    x_ph, y_ph, signed_grad = build_fgsm_op(model)

    out_root = os.path.join('data', name_tag)
    train_dir = os.path.join(out_root, 'train')
    test_dir = os.path.join(out_root, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    np.save(os.path.join(train_dir, 'label.npy'), y_train)
    np.save(os.path.join(test_dir, 'label.npy'), y_test)

    for eps in eps_list:
        print(f'Generating FGSM eps={eps} ...')
        adv_train = fgsm_generate(model, x_train, y_train, eps, x_ph, y_ph, signed_grad, clip_min, clip_max)
        adv_test = fgsm_generate(model, x_test, y_test, eps, x_ph, y_ph, signed_grad, clip_min, clip_max)

        np.save(os.path.join(train_dir, f'epsilon{eps}.npy'), adv_train)
        np.save(os.path.join(test_dir, f'epsilon{eps}.npy'), adv_test)

    meta = {
        'model_name': model_name,
        'name_tag': name_tag,
        'seed': seed,
        'train_ratio': train_ratio,
        'subset_ratio': subset_ratio,
        'eps_list': eps_list,
        'clip_min': clip_min,
        'clip_max': clip_max,
        'train_size': int(len(x_train)),
        'test_size': int(len(x_test)),
    }
    with open(os.path.join(out_root, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    print(f'Saved adversarial data to: {out_root}')


if __name__ == '__main__':
    main()
