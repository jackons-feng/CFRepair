import numpy as np
import tensorflow as tf
import tensorflow.python.keras.backend as K


class MTLBase:
    mode = 'grad'

    def __init__(self, optimizer, reduction='sum'):
        self._optim = optimizer
        self._reduction = reduction

    def _flatten_grads(self, task_grads, vars_to_update):
        flat = []
        for g, v in zip(task_grads, vars_to_update):
            if g is None:
                g = tf.zeros_like(v)
            flat.append(tf.reshape(g, [-1]))
        return tf.concat(flat, axis=0)

    def _reshape_flat_grads(self, merged_grad_flat, vars_to_update):
        new_grads = []
        curr_pos = 0
        for v in vars_to_update:
            v_size = np.prod(v.shape.as_list())
            g_part = tf.reshape(merged_grad_flat[curr_pos:curr_pos + v_size], v.shape)
            new_grads.append(g_part)
            curr_pos += v_size
        return new_grads

    def _stack_task_grads(self, grads_per_task, vars_to_update):
        return tf.stack([
            self._flatten_grads(task_grads, vars_to_update) for task_grads in grads_per_task
        ], axis=0)

    def _project_to_simplex(self, v):
        v_sorted = tf.sort(v, direction='DESCENDING')
        cssv = tf.cumsum(v_sorted) - 1.0
        idx = tf.cast(tf.range(1, tf.shape(v)[0] + 1), v.dtype)
        cond = v_sorted - cssv / idx > 0
        rho = tf.reduce_max(tf.where(cond, idx, tf.zeros_like(idx)))
        rho_idx = tf.cast(rho - 1, tf.int32)
        theta = cssv[rho_idx] / rho
        return tf.maximum(v - theta, 0.0)

    def _simplex_qp(self, gram, linear=None, steps=50, lr=0.1):
        num_tasks = tf.shape(gram)[0]
        w = tf.fill([num_tasks], 1.0 / tf.cast(num_tasks, tf.float32))
        linear = tf.zeros_like(w) if linear is None else linear

        def body(i, w_cur):
            grad = tf.linalg.matvec(gram + tf.transpose(gram), w_cur) + linear
            w_next = self._project_to_simplex(w_cur - lr * grad)
            return i + 1, w_next

        def cond(i, _w_cur):
            return i < steps

        _, w_final = tf.while_loop(cond, body, [0, w])
        return w_final


class PCGrad(MTLBase):
    """
    Projected Conflicting Gradients (PCGrad) implementation.
    Reference: https://arxiv.org/abs/2001.06782
    """
    mode = 'grad'

    def _project_conflicting(self, grads, vars_to_update):
        """
        grads: list of lists of gradients (one list per task)
        vars_to_update: list of variables being optimized
        """
        shared_grads = [self._flatten_grads(task_grads, vars_to_update) for task_grads in grads]
        num_tasks = len(shared_grads)

        # Shuffle task order (note: fixed per graph build in TF1 graph mode)
        task_indices = np.arange(num_tasks)
        np.random.shuffle(task_indices)

        projected_grads = [g for g in shared_grads]

        for i in task_indices:
            other_indices = [j for j in task_indices if i != j]
            np.random.shuffle(other_indices)
            for j in other_indices:
                inner_prod = tf.reduce_sum(projected_grads[i] * shared_grads[j])
                norm_sq = tf.reduce_sum(shared_grads[j] * shared_grads[j]) + 1e-8
                projected_grads[i] = tf.cond(
                    inner_prod < 0,
                    lambda: projected_grads[i] - (inner_prod / norm_sq) * shared_grads[j],
                    lambda: projected_grads[i]
                )

        if self._reduction == 'sum':
            merged_grad = tf.reduce_sum(projected_grads, axis=0)
        else:
            merged_grad = tf.reduce_mean(projected_grads, axis=0)

        return merged_grad

    def apply(self, losses, grads_per_task, vars_to_update):
        merged_grad_flat = self._project_conflicting(grads_per_task, vars_to_update)
        new_grads = self._reshape_flat_grads(merged_grad_flat, vars_to_update)
        return self._optim.apply_gradients(zip(new_grads, vars_to_update))


class MGDA(MTLBase):
    """
    MGDA for multiple tasks via simplex-constrained QP.
    Reference: https://arxiv.org/abs/1810.04650
    """
    mode = 'grad'

    def __init__(self, optimizer, reduction='sum', qp_steps=50, qp_lr=0.1, two_task_closed_form=True):
        super().__init__(optimizer, reduction)
        self._qp_steps = qp_steps
        self._qp_lr = qp_lr
        self._two_task_closed_form = two_task_closed_form

    def _two_task_weights(self, g1, g2):
        g_diff = g1 - g2
        denom = tf.reduce_sum(g_diff * g_diff) + 1e-8
        w1 = tf.reduce_sum(g2 * g_diff) / denom
        w1 = tf.clip_by_value(w1, 0.0, 1.0)
        w2 = 1.0 - w1
        return tf.stack([w1, w2], axis=0)

    def apply(self, losses, grads_per_task, vars_to_update):
        g_stack = self._stack_task_grads(grads_per_task, vars_to_update)
        num_tasks = tf.shape(g_stack)[0]

        def two_task():
            w = self._two_task_weights(g_stack[0], g_stack[1])
            return w

        def multi_task():
            gram = tf.matmul(g_stack, g_stack, transpose_b=True)
            return self._simplex_qp(gram, steps=self._qp_steps, lr=self._qp_lr)

        w = tf.cond(
            tf.logical_and(self._two_task_closed_form, tf.equal(num_tasks, 2)),
            two_task,
            multi_task
        )

        merged_grad_flat = tf.linalg.matvec(tf.transpose(g_stack), w)
        new_grads = self._reshape_flat_grads(merged_grad_flat, vars_to_update)
        return self._optim.apply_gradients(zip(new_grads, vars_to_update))


class CAGrad(MTLBase):
    """
    CAGrad (Conflict-Averse Gradient Descent).
    Reference: https://arxiv.org/abs/2110.14048
    """
    mode = 'grad'

    def __init__(self, optimizer, reduction='sum', c=0.5, steps=20, lr=0.1):
        super().__init__(optimizer, reduction)
        self._c = c
        self._steps = steps
        self._lr = lr

    def _cagrad_weights(self, gram, g0_norm):
        num_tasks = tf.shape(gram)[0]
        w = tf.fill([num_tasks], 1.0 / tf.cast(num_tasks, tf.float32))

        Gg = tf.reduce_mean(gram, axis=1)
        c = self._c * g0_norm

        def body(i, w_cur):
            gw = tf.linalg.matvec(gram, w_cur)
            gw_norm = tf.norm(gw) + 1e-8
            grad = Gg + c * (gw / gw_norm)
            w_next = self._project_to_simplex(w_cur - self._lr * grad)
            return i + 1, w_next

        def cond(i, _w_cur):
            return i < self._steps

        _, w_final = tf.while_loop(cond, body, [0, w])
        return w_final

    def apply(self, losses, grads_per_task, vars_to_update):
        g_stack = self._stack_task_grads(grads_per_task, vars_to_update)
        num_tasks = tf.shape(g_stack)[0]

        gram = tf.matmul(g_stack, g_stack, transpose_b=True)
        g0 = tf.reduce_mean(g_stack, axis=0)
        g0_norm = tf.norm(g0) + 1e-8

        w = self._cagrad_weights(gram, g0_norm)
        gw = tf.linalg.matvec(tf.transpose(g_stack), w)
        gw_norm = tf.norm(gw) + 1e-8

        lmbda = (self._c * g0_norm) / gw_norm
        g = (g0 + lmbda * gw) / (1.0 + self._c)

        new_grads = self._reshape_flat_grads(g, vars_to_update)
        return self._optim.apply_gradients(zip(new_grads, vars_to_update))


class NashMTL(MTLBase):
    """
    Nash-MTL implementation (fixed-point solver for alpha).
    Reference: https://arxiv.org/abs/2203.11074
    """
    mode = 'grad'

    def __init__(self, optimizer, reduction='sum', steps=50, eps=1e-8, tol=1e-4):
        super().__init__(optimizer, reduction)
        self._steps = steps
        self._eps = eps
        self._tol = tol

    def _solve_alpha(self, gtg):
        num_tasks = tf.shape(gtg)[0]
        alpha = tf.fill([num_tasks], 1.0 / tf.cast(num_tasks, tf.float32))

        def body(i, a_cur):
            gtg_a = tf.linalg.matvec(gtg, a_cur)
            a_next = a_cur / (gtg_a + self._eps)                     # 关键修改：乘以 a_cur
            a_next = a_next * (tf.cast(num_tasks, tf.float32) / (tf.reduce_sum(a_next) + self._eps))
            return i + 1, a_next

        def cond(i, a_cur):
            gtg_a = tf.linalg.matvec(gtg, a_cur)
            a_next = a_cur / (gtg_a + self._eps)
            a_next = a_next * (tf.cast(num_tasks, tf.float32) / (tf.reduce_sum(a_next) + self._eps))
            return tf.logical_and(i < self._steps, tf.norm(a_next - a_cur) > self._tol)

        _, alpha_final = tf.while_loop(cond, body, [0, alpha])
        return alpha_final

    def apply(self, losses, grads_per_task, vars_to_update):
        g_stack = self._stack_task_grads(grads_per_task, vars_to_update)
        gtg = tf.matmul(g_stack, g_stack, transpose_b=True)
        alpha = self._solve_alpha(gtg)
        merged_grad_flat = tf.linalg.matvec(tf.transpose(g_stack), alpha)
        new_grads = self._reshape_flat_grads(merged_grad_flat, vars_to_update)
        return self._optim.apply_gradients(zip(new_grads, vars_to_update))


class GradNorm(MTLBase):
    """
    GradNorm loss balancing (ICML 2018).
    This version maintains per-task weights externally.
    Reference: https://arxiv.org/abs/1711.02257
    """
    mode = 'loss'
    uses_loss_weights = True
    normalize_loss_weights = True

    def __init__(
        self,
        optimizer,
        reduction='sum',
        alpha=1.0,
        init_losses=None
    ):
        super().__init__(optimizer, reduction)
        self._alpha = alpha
        self._init_losses = init_losses

    def apply(self, losses, grads_per_task, vars_to_update, loss_weights):
        """
        losses: list of task losses (scalars)
        grads_per_task: list of lists of gradients (one list per task) for all trainable vars
        vars_to_update: list of variables being optimized (network parameters)
        loss_weights: trainable variable of shape (num_tasks,) representing w_i
        """
        if loss_weights is None:
            raise ValueError("GradNorm requires loss_weights variable.")

        # 1. 记录初始损失（仅在第一次调用时设置）
        if self._init_losses is None:
            self._init_losses = [tf.stop_gradient(l) for l in losses]

        # 2. 计算加权总损失
        weighted_losses = [loss_weights[i] * losses[i] for i in range(len(losses))]
        total_loss = tf.add_n(weighted_losses)

        # 3. 计算每个任务的加权梯度范数
        task_grad_norms = []
        for i, task_grads in enumerate(grads_per_task):
            # 原始梯度展平
            flat_grad = self._flatten_grads(task_grads, vars_to_update)   # ∇_W L_i
            # 加权梯度 = w_i * ∇_W L_i
            weighted_flat_grad = loss_weights[i] * flat_grad
            task_grad_norms.append(tf.norm(weighted_flat_grad))

        avg_grad_norm = tf.reduce_mean(task_grad_norms)

        # 4. 计算相对损失下降率 r_i(t) = L_i(t) / L_i(0)
        loss_ratios = [losses[i] / (init_loss + 1e-8) for i, init_loss in enumerate(self._init_losses)]

        # 5. 计算目标梯度范数
        target_norms = [avg_grad_norm * (ratio ** self._alpha) for ratio in loss_ratios]

        # 6. GradNorm 损失：L1 距离
        gradnorm_loss = tf.add_n([tf.abs(g - t) for g, t in zip(task_grad_norms, target_norms)])

        # 返回总损失（用于更新网络参数）和 GradNorm 损失（用于更新 w_i）
        return total_loss, gradnorm_loss


class RLW(MTLBase):
    """
    Random Loss Weighting (RLW) [2022].
    Reference: https://arxiv.org/pdf/2111.10603.pdf
    """
    mode = 'loss'
    uses_loss_weights = False
    normalize_loss_weights = False

    def __init__(self, optimizer, reduction='sum'):
        super().__init__(optimizer, reduction)
        self.last_weights = None

    def apply(self, losses, grads_per_task, vars_to_update, loss_weights=None):
        num_tasks = len(losses)
        raw = tf.random.normal([num_tasks])
        weights = tf.nn.softmax(raw) * tf.cast(num_tasks, tf.float32)
        self.last_weights = weights
        total_loss = tf.add_n([w * l for w, l in zip(tf.unstack(weights), losses)])
        return total_loss, None


class DWA(MTLBase):
    """
    Dynamic Weight Average (DWA, CVPR 2019).
    Computes weights from relative loss descent between iterations.
    """
    mode = 'loss'
    uses_loss_weights = False
    normalize_loss_weights = False

    def __init__(self, optimizer, reduction='sum', temp=2.0):
        super().__init__(optimizer, reduction)
        self._temp = temp
        self._prev_losses = None
        self._prev_prev_losses = None
        self.last_weights = None

    def apply(self, losses, grads_per_task, vars_to_update, loss_weights=None):
        num_tasks = len(losses)

        if self._prev_losses is None or self._prev_prev_losses is None:
            weights = tf.ones([num_tasks], dtype=tf.float32)
        else:
            ratios = tf.stack([
                self._prev_losses[i] / (self._prev_prev_losses[i] + 1e-8)
                for i in range(num_tasks)
            ])
            logits = ratios / self._temp
            weights = tf.nn.softmax(logits) * tf.cast(num_tasks, tf.float32)

        self.last_weights = weights
        total_loss = tf.add_n([w * l for w, l in zip(tf.unstack(weights), losses)])

        self._prev_prev_losses = self._prev_losses
        self._prev_losses = [tf.stop_gradient(l) for l in losses]

        return total_loss, None
