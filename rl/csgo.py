"""
Implementation of clip and keep.

There are two versions here, one as a clipping algorithm, the other as a full-blown optimizer (WIP)
"""


import numpy as np
import torch
from torch import Tensor
from torch.optim import Optimizer
from typing import List, Optional


@torch.no_grad()
def clip_and_keep(
    optimizer,
    mode:str = "mode1",
    alpha = 0.99,
    decay:float = 1.0,
    c1 = 5,
    c2 = None,
    clip_scaled_gradients: bool = False,
):
    """
    Implementation of clip and keep as a clipping algorithm

    @optimizer: The optimizer that provides the parameters to clip. Accumulator will be stored with this optimizer
    @mode: the clipping mode to use [global_norm, co-ordinate, mode1, mode2, mode3, mode4]

    For some input gradient G, and accumulator A

    mode1: g = clip(G+alpha*A), r = residual(G+alpha*R); A = A * (1-alpha) + r (set alpha=1 for MH version)
    mode2: g = clip(G)+alpha*A, r = residual(G); A = A * (1-alpha) + r (don't clip residual component)
    mode3: g = clip(G)+clip(A), A = residual(G)+residual(A), (gradient and accumulator are clipped independently,
                using two different clipping constants)

    Mode2 can work quite well, but requires better tuning of alpha.

    @param alpha: how much of the accumulator to apply at each step
    @param decay: Residual accumulator (A) decays by this each step. Introduces bias, but might help in some cases.
    @param clip_scaled_gradients: If true scales gradients then clips them, then unscales (only works for adam)

    @returns stats (gradient and accumulator are modified within the supplied optimizer)

    """

    clipped = 0
    count = 1
    acc_norms = []
    grad_norms = []

    def clip_and_remainder(x, c):
        clipped = torch.clip(x, -c, c)
        remainder = x - clipped
        return clipped, remainder

    if type(optimizer) is torch.optim.Adam and optimizer.state == {}:
        # adam uses lazy state initialization, if we add something to the state the initialization will fail,
        # so we have to run one step here to initialize it. This does mean taking teh first step twice.
        optimizer.step()

    for pg in optimizer.param_groups:
        for p in pg['params']:

            if p.grad is None:
                continue

            G = p.grad.data
            if 'acc' in optimizer.state[p]:
                A = optimizer.state[p]['acc']
            else:
                A = torch.zeros_like(G)

            grad_norms.append((torch.square(G).sum() ** 0.5).cpu())
            count += np.prod(G.shape)

            # apply scale correction
            if clip_scaled_gradients:
                # we do this by modifying the clipping values
                # this has the advantage that we don't store large gradients in the accumlator, that later get
                # `paid back' incorrectly due to scaling change. Ideally we'd integrate this all into an optimizer,
                # which I will do later on.

                assert type(optimizer) is torch.optim.Adam, "Scaled gradients only works with Adam at the moment"
                assert 'exp_avg_sq' in optimizer.state[p], "Could not found second moment information in optimizer"

                bias_correction = 1 - (pg["betas"][1] ** optimizer.state[p]['step'])
                scale = (optimizer.state[p]['exp_avg_sq'] / bias_correction).sqrt() + pg['eps']
                _c1 = scale * c1
                _c2 = scale * c2 if c2 is not None else None
            else:
                _c1 = c1
                _c2 = c2

            if mode == "mode1":
                # clip the gradient + accumulator
                g, r = clip_and_remainder(G + alpha * A, _c1)
                A = A * (1 - alpha) + r
            elif mode == "mode2":
                # clip the gradient, but not the accumulator
                g, r = clip_and_remainder(G, _c1)
                g += alpha * A
                A = A * (1 - alpha) + r
            elif mode == "mode3":
                # clip both independently
                g_1, r_1 = clip_and_remainder(G, _c1)
                g_2, r_2 = clip_and_remainder(A, _c2)
                g = g_1 + g_2
                r = r_1 + r_2
                A = r

            # elif mode == "cak2":
            #     # clip(r+g)
            #     g += A * args.csgo_friction
            #     A *= (1 - args.csgo_friction)
            #     clipped += torch.gt(torch.abs(g), args.max_grad_norm).sum()
            #     head = torch.clip(g, -args.max_grad_norm, +args.max_grad_norm)
            #     tail = g - head
            #     A += tail
            #     p.grad.data = head
            # elif mode == "cak3":
            #     # works if csgo_friction is tuned really well
            #     # clipped(g)+r
            #     clipped += torch.gt(torch.abs(g), args.max_grad_norm).sum()
            #     head = torch.clip(g, -args.max_grad_norm, +args.max_grad_norm)
            #     tail = g - head
            #     p.grad.data = head + A * args.csgo_friction
            #     A *= (1 - args.csgo_friction) * args.csgo_decay
            #     A += tail
            # elif mode == "cak4":
            #     # clip(g)+r on scaled gradients.
            #     assert args.optimizer == "adam", "Only adam supported with this clipping method."
            #     if 'exp_avg_sq' not in optimizer.state[p]:
            #         continue
            #     bias_correction = 1 - (args.adam_beta2 ** optimizer.state[p]['step'])
            #     scale = (optimizer.state[p]['exp_avg_sq'] / bias_correction).sqrt() + args.adam_epsilon
            #     clipped += torch.gt(torch.abs(g / scale), args.max_grad_norm).sum()
            #     head = torch.clip(g / scale, -args.max_grad_norm, +args.max_grad_norm) * scale
            #     tail = g - head
            #     p.grad.data = head + A * args.csgo_friction
            #     A *= (1 - args.csgo_friction) * args.csgo_decay
            #     A += tail
            # elif mode == "cak5":
            #     # clip(g+r) on scaled gradients.
            #     assert args.optimizer == "adam", "Only adam supported with this clipping method."
            #     if 'exp_avg_sq' not in optimizer.state[p]:
            #         continue
            #     bias_correction = 1 - (args.adam_beta2 ** optimizer.state[p]['step'])
            #     scale = (optimizer.state[p]['exp_avg_sq'] / bias_correction).sqrt() + args.adam_epsilon
            #     acc_norms[-1] = ((torch.square(
            #         A * scale).sum() * args.csgo_friction) ** 0.5).cpu()  # override a2 here...
            #     g = (g / scale) + A * args.csgo_friction
            #     clipped += torch.gt(torch.abs(g), args.max_grad_norm).sum()
            #     head = torch.clip(g, -args.max_grad_norm, +args.max_grad_norm) * scale
            #     tail = g - head
            #     p.grad.data = head
            #     p.grad.data *= scale
            #     A *= (1 - args.csgo_friction) * args.csgo_decay
            #     A += tail
            else:
                raise ValueError("Invalid clip_mode.")

            clipped += int(torch.gt(torch.abs(r), 1e-6).sum().cpu())
            acc_norms.append((torch.square(A).sum() ** 0.5).cpu())

            # update grads (I should probably copy these not replace them?)
            p.grad.data = g

            # save accumulator
            optimizer.state[p]['acc'] = (A * decay)

    if clipped > 0:
        pass

    return {
        'clip': float(clipped / count),
        "a2": np.asarray(acc_norms),
        "g2": np.asarray(grad_norms),
    }


class CSGO(Optimizer):
    """
        Implements clipped stochastic gradient descent (optionally with momentum).
    """

    def __init__(self, params, lr=None, friction=0.01, clip=4.00, decay=0.99):
        if lr is not None and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(lr=lr, clip=clip, friction=friction, decay=decay)
        super(CSGO, self).__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)

    @torch.no_grad()
    def step(self):
        """
        Performs a single optimization step.
        """
        loss = None

        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            square_averages_list = []

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)
                    if p.grad.is_sparse:
                        raise Exception("Sparse gradient not supported.")

                    state = self.state[p]

                    if 'momentum_buffer' not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state['momentum_buffer'])

                    if 'square_buffer' not in state:
                        square_averages_list.append(None)
                    else:
                        square_averages_list.append(state['square_buffer'])

            sgd(params_with_grad,
                d_p_list,
                momentum_buffer_list,
                square_averages_list,
                lr = group['lr'],
                clip = group['clip'],
                friction = group['friction'],
                decay = group['decay'],
            )

            # update momentum_buffers in state
            for p, momentum_buffer, square_buffer in zip(params_with_grad, momentum_buffer_list, square_averages_list):
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer
                state['square_buffer'] = square_buffer
                # debugging...
                scaling = 1/(square_buffer+1e-5)
                print(f"{str(p.shape):<40} {float(momentum_buffer.mean()):<10.3f} {float(momentum_buffer.std()):<10.3f} {float(scaling.mean()):<10.2f} {float(scaling.std()):<10.2f}")

        return loss


def sgd(params: List[Tensor],
        d_p_list: List[Tensor],
        momentum_buffer_list: List[Optional[Tensor]],
        square_avgs: List[Optional[Tensor]],
        lr: float,
        clip: float,
        friction: float,
        decay: float,
    ):
    r"""Functional API that performs SGD algorithm computation.

    See :class:`~torch.optim.SGD` for details.
    """

    _single_tensor_sgd(
        params,
        d_p_list,
        momentum_buffer_list,
        square_avgs,
        lr=lr,
        clip=clip,
        friction=friction,
        decay=decay,
        )


def _single_tensor_sgd(params: List[Tensor],
                       d_p_list: List[Tensor],
                       momentum_buffer_list: List[Optional[Tensor]],
                       square_buffer_list: List[Tensor],
                       lr: float,
                       clip: float,
                       friction: float = 0.01,
                       decay: float = 1.0,
                       ):

    for i, param in enumerate(params):

        square_avg = square_buffer_list[i]
        if square_avg is None:
            square_avg = torch.zeros_like(param, requires_grad=False)
            square_buffer_list[i] = square_avg

        correction_step = d_p_list[i]

        beta = 0.99
        eps = 1e-5
        square_avg.mul_(beta).addcmul_(correction_step, correction_step, value=1-beta)
        avg = square_avg.sqrt().add_(eps)

        correction_step /= avg

        g_head = torch.clip(correction_step, -clip, clip)
        g_tail = correction_step - g_head

        accumulator = momentum_buffer_list[i]

        if accumulator is None:
            accumulator = torch.clone(g_tail).detach()
            momentum_buffer_list[i] = accumulator
        else:
            accumulator += g_tail

        # clipped correction step.
        param -= lr * g_head

        # would be better to transfer up to some fixed length? not 0.01
        # we already know next momentum step, so apply it now
        delta = accumulator * friction
        param -= delta * lr
        accumulator -= delta
        accumulator *= decay
