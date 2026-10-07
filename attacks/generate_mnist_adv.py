import os
import sys
import json
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import tensorflow.python.keras.backend as K
from tensorflow.keras.models import load_model


BATCH_SIZE = 128


def parse_eps_list(s):
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def setup_tf(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)

    tf.compat.v1.disable_eager_execution()
    gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
    sess = tf.compat.v1.Session(
        config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)
    )
    K.set_session(sess)


def load_mnist_clean_data():
    (x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()
    x_train = np.expand_dims(x_train.astype("float32") / 255.0, axis=-1)
    y_train = y_train.astype("int64")
    return x_train, y_train


def build_fgsm_op(model):
    x_ph = tf.compat.v1.placeholder(
        tf.float32,
        shape=(None,) + model.input_shape[1:],
        name="x_ph"
    )
    y_ph = tf.compat.v1.placeholder(
        tf.int32,
        shape=(None,),
        name="y_ph"
    )

    output = model(x_ph)
    num_classes = model.output_shape[-1]
    y_onehot = tf.one_hot(y_ph, num_classes)

    try:
        last_activation = getattr(model.layers[-1], "activation", None)
        is_softmax = (
            last_activation is not None
            and last_activation.__name__ == "softmax"
        )
    except Exception:
        is_softmax = False

    if is_softmax:
        loss_vec = tf.keras.losses.categorical_crossentropy(
            y_onehot,
            output,
            from_logits=False
        )
        loss = tf.reduce_mean(loss_vec)
    else:
        loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                labels=y_onehot,
                logits=output
            )
        )

    grad = tf.gradients(loss, x_ph)[0]
    signed_grad = tf.sign(grad)

    return x_ph, y_ph, signed_grad


def fgsm_generate(model, x, y, epsilon, x_ph, y_ph, signed_grad):
    sess = K.get_session()
    out = []

    n_batches = int(np.ceil(len(x) / BATCH_SIZE))

    for b in range(n_batches):
        start = b * BATCH_SIZE
        end = min((b + 1) * BATCH_SIZE, len(x))

        xb = x[start:end]
        yb = y[start:end]

        g = sess.run(
            signed_grad,
            feed_dict={
                x_ph: xb,
                y_ph: yb
            }
        )

        adv = xb + epsilon * g

        # MNIST 输入范围是 [0, 1]
        adv = np.clip(adv, 0.0, 1.0)

        out.append(adv.astype("float32"))

    return np.concatenate(out, axis=0)


def predict_labels(model, x):
    preds = model.predict(x, batch_size=BATCH_SIZE)
    return np.argmax(preds, axis=1)


def save_split_adv(model, split_name, x, y, eps_list, out_root, x_ph, y_ph, signed_grad):
    split_dir = os.path.join(out_root, split_name)
    os.makedirs(split_dir, exist_ok=True)

    np.save(os.path.join(split_dir, "label.npy"), y)

    # 修复阶段如果需要从数据集文件夹拿 clean 数据，可以直接用这个
    np.save(os.path.join(split_dir, "clean.npy"), x)

    results = {}

    for eps in eps_list:
        print(f"Generating {split_name} FGSM eps={eps} ...")

        x_adv = fgsm_generate(
            model=model,
            x=x,
            y=y,
            epsilon=eps,
            x_ph=x_ph,
            y_ph=y_ph,
            signed_grad=signed_grad
        )

        y_pred = predict_labels(model, x_adv)
        adv_error = float(np.mean(y_pred != y))

        save_path = os.path.join(split_dir, f"epsilon{eps}.npy")
        np.save(save_path, x_adv)

        print(f"Saved: {save_path}")
        print(f"{split_name} adv error eps={eps}: {adv_error:.6f}")

        results[str(eps)] = {
            "adv_error_rate": adv_error,
            "save_path": save_path
        }

    return results


def main():
    if len(sys.argv) > 5:
        print("Usage: python generate_mnist_adv.py [seed] [train_ratio] [eps_list] [subset_ratio]")
        print("Example: python generate_mnist_adv.py 42 0.5 0.05,0.1,0.2 1.0")
        sys.exit(1)

    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    train_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    eps_list = parse_eps_list(sys.argv[3]) if len(sys.argv) > 3 else [0.05, 0.1, 0.2]
    subset_ratio = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0

    if train_ratio <= 0 or train_ratio >= 1:
        raise ValueError("train_ratio must be in (0, 1).")

    if subset_ratio <= 0 or subset_ratio > 1:
        raise ValueError("subset_ratio must be in (0, 1].")

    model_name = "mnist"
    model_path = "../model/mnist.h5"
    out_root = os.path.join("data", model_name)

    setup_tf(seed)

    print(f"Loading MNIST backdoor model: {model_path}")
    model = load_model(model_path, compile=False)

    print("Loading MNIST clean data from tf.keras.datasets.mnist")
    x_clean, y_clean = load_mnist_clean_data()

    print(f"Original clean shape: {x_clean.shape}")
    print(f"Original label shape: {y_clean.shape}")
    print(f"Original range: [{x_clean.min():.6f}, {x_clean.max():.6f}]")

    # 注意：
    # 不随机打乱，不 stratified split。
    # 因为你不改 loc_neurons.py，它会重新 load_clean_data('mnist')，
    # 然后直接用 x_clean[:n_align] 和 data/mnist/train/epsilon.npy 对齐。
    n_total = int(len(x_clean) * subset_ratio)
    x_clean = x_clean[:n_total]
    y_clean = y_clean[:n_total]

    n_train = int(len(x_clean) * train_ratio)

    x_train = x_clean[:n_train]
    y_train = y_clean[:n_train]

    x_test = x_clean[n_train:]
    y_test = y_clean[n_train:]

    print(f"Train shape: {x_train.shape}, labels: {y_train.shape}")
    print(f"Test shape:  {x_test.shape}, labels: {y_test.shape}")

    x_ph, y_ph, signed_grad = build_fgsm_op(model)

    train_results = save_split_adv(
        model=model,
        split_name="train",
        x=x_train,
        y=y_train,
        eps_list=eps_list,
        out_root=out_root,
        x_ph=x_ph,
        y_ph=y_ph,
        signed_grad=signed_grad
    )

    test_results = save_split_adv(
        model=model,
        split_name="test",
        x=x_test,
        y=y_test,
        eps_list=eps_list,
        out_root=out_root,
        x_ph=x_ph,
        y_ph=y_ph,
        signed_grad=signed_grad
    )

    meta = {
        "model_name": model_name,
        "model_path": model_path,
        "seed": seed,
        "train_ratio": train_ratio,
        "subset_ratio": subset_ratio,
        "eps_list": eps_list,
        "train_size": int(len(x_train)),
        "test_size": int(len(x_test)),
        "input_shape": list(x_train.shape[1:]),
        "input_min": float(x_clean.min()),
        "input_max": float(x_clean.max()),
        "save_root": out_root,
        "important_note": (
            "generate_mnist_adv.py does not depend on DATASET_CONFIG. "
            "The train split is a prefix of MNIST clean data to keep compatibility "
            "with unchanged loc_neurons.py."
        ),
        "train_results": train_results,
        "test_results": test_results
    }

    with open(os.path.join(out_root, "meta_adv.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\nDone.")
    print(f"Saved adversarial data to: {out_root}")

    if eps_list:
        print("\nNext localization command:")
        print(f"python loc_neurons.py mnist 1000 {eps_list[0]} {seed}")


if __name__ == "__main__":
    main()