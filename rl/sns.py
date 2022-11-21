"""
Handles noise estimation.
"""

from config import BaseConfig, args
from rollout import Runner
from . import utils

import torch
import numpy as np

from collections import deque
import math
import ast
import time as clock


class SimpleNoiseScaleConfig(BaseConfig):
    """
    Config settings for simple noise scale
    """
    enabled:bool = False # Enables generation of simple noise scale estimates.
    labels: str = ['policy','distil','value', 'value_heads']  # value|value_heads|distil|policy
    period: int = 3  # Generate estimates every n updates.
    max_heads: int = 7  # Limit to this number of heads when doing per head noise estimate.
    b_big: int = 2048
    b_small: int = 128
    fake_noise: bool = False # Replaces value_head gradient with noise based on horizon.
    smoothing_mode: str = "ema" # ema|avg
    smoothing_horizon_avg: int = 1e6, # how big to make averaging window
    smoothing_horizon_s: int = 0.2e6, # how much to smooth s
    smoothing_horizon_g2: int = 1.0e6, # how much to smooth g2
    smoothing_horizon_policy: int = 5e6, # how much to smooth g2 for policy (normally much higher)

    def __init__(self, parser):
        super().__init__(prefix="sns", parser=parser)


def get_ema_constant(self, required_horizon: int, updates_every: int = 1):
    """
    Returns an ema coefficent to match given horizon (in environment interactions), when updates will be applied
    every "updates_every" rollouts
    """
    if required_horizon == 0:
        return 0
    updates_every_steps = (updates_every * self.N * self.A)
    ema_horizon = required_horizon / updates_every_steps
    return 1 - (1 / ema_horizon)

def process_noise_scale(
        runner: Runner,
        g_b_small_squared: float,
        g_b_big_squared: float,
        label: str,
        verbose: bool = True,
        b_small: int = None,
        b_big: int = None,

):
    """
    Logs noise levels using provided gradients
    """

    b_small = b_small or args.sns.b_small
    b_big = b_big or args.sns.b_big

    est_s = (g_b_small_squared - g_b_big_squared) / (1 / b_small - 1 / b_big)
    est_g2 = (b_big * g_b_big_squared - b_small * g_b_small_squared) / (b_big - b_small)

    if args.sns.smoothing_mode == "avg":
        # add these samples to the mix
        for var_name, var_value in zip(['s', 'g2'], [est_s, est_g2]):
            if f'{label}_{var_name}_history' not in runner.noise_stats:
                history_frames = int(args.sns.smoothing_horizon_avg)  # 5 million frames should be about right
                ideal_updates_length = history_frames / (runner.N * runner.A)
                buffer_len = int(max(10, math.ceil(ideal_updates_length / args.sns.period)))
                runner.noise_stats[f'{label}_{var_name}_history'] = deque(maxlen=buffer_len)
            runner.noise_stats[f'{label}_{var_name}_history'].append(var_value)
            runner.noise_stats[f'{label}_{var_name}'] = np.mean(runner.noise_stats[f'{label}_{var_name}_history'])
    elif args.sns.smoothing_mode == "ema":
        ema_s = get_ema_constant(args.sns.smoothing_horizon_s, args.sns.period)
        g2_horizon = args.sns.smoothing_horizon_policy if label == "policy" else args.sns.smoothing_horizon_g2
        ema_g2 = get_ema_constant(g2_horizon, args.sns.period)
        # question: we do we need to smooth both of these? which is more noisy? I think it's just g2 right?
        utils.dictionary_ema(runner.noise_stats, f'{label}_s', est_s, ema_s)
        utils.dictionary_ema(runner.noise_stats, f'{label}_g2', est_g2, ema_g2)
    else:
        raise ValueError(f"Invalid smoothing mode {args.sns.smoothing_mode}.")

    smooth_s = float(runner.noise_stats[f'{label}_s'])
    smooth_g2 = float(runner.noise_stats[f'{label}_g2'])

    # g2 estimate is frequently negative. If ema average bounces below 0 the ratio will become negative.
    # to avoid this we clip the *smoothed* g2 to epsilon.
    # alternative include larger batch_sizes, and / or larger EMA horizon.
    # noise levels above 1000 will not be very accurate, but around 20 should be fine.
    epsilon = 1e-4  # we can't really measure noise above this level anyway (which is around a ratio of 10k:1)
    ratio = smooth_s / (max(0.0, smooth_g2) + epsilon)

    runner.noise_stats[f'{label}_ratio'] = ratio
    if 'head' in label:
        # keep track of which heads we have results for
        try:
            idx = int(label.split("_")[-1])
            if 'active_heads' not in runner.noise_stats:
                runner.noise_stats['active_heads'] = set()
            runner.noise_stats['active_heads'].add(idx)
        except:
            # this is fine
            pass

    # maybe this is too much logging?
    runner.log.watch(f'sns_{label}_smooth_s', smooth_s, display_precision=0, display_width=0,
                     display_name=f"sns_{label}_s")
    runner.log.watch(f'sns_{label}_smooth_g2', smooth_g2, display_precision=0, display_width=0,
                     display_name=f"sns_{label}_g2")
    runner.log.watch(f'sns_{label}_s', est_s, display_precision=0, display_width=0)
    runner.log.watch(f'sns_{label}_g2', est_g2, display_precision=0, display_width=0)
    runner.log.watch(f'sns_{label}_b', ratio, display_precision=0, display_width=0)
    runner.log.watch(
        f'sns_{label}',
        np.clip(ratio, 0, float('inf')) ** 0.5,
        display_precision=0,
        display_width=8 if verbose else 0,
    )

    return runner.noise_stats[f'{label}_ratio']


def estimate_noise_scale(
        runner: Runner,
        batch_data,
        mini_batch_func,
        optimizer: torch.optim.Optimizer,
        label,
        verbose: bool = True,
):
    """
    Estimates the critical batch size using the gradient magnitude of a small batch and a large batch

    ema smoothing produces cleaner results, but is biased.

    new version...

    See: https://arxiv.org/pdf/1812.06162.pdf
    """

    b_small = args.sns.b_small

    if label == "policy":
        # always use full batch for policy (it's required to get the precision needed)
        b_big = runner.N * runner.A
    else:
        b_big = args.sns.b_big

    # resample data
    # this also shuffles order
    data = {}
    samples = np.random.choice(range(len(batch_data["prev_state"])), b_big, replace=False)
    for k, v in batch_data.items():
        if k.startswith('*'):
            # these inputs are uploaded directly, and not sampled down.
            data[k] = batch_data[k]
        else:
            data[k] = batch_data[k][samples]

    assert b_big % b_small == 0, "b_small must divide b_big"
    mini_batches = b_big // b_small

    small_norms_sqr = []
    big_grad = None

    for i in range(mini_batches):
        optimizer.zero_grad(set_to_none=True)
        segment = slice(i * b_small, (i + 1) * b_small)
        mini_batch_data = {}
        for k, v in data.items():
            mini_batch_data[k] = data[k][segment]
            if type(mini_batch_data[k]) is np.ndarray:
                if mini_batch_data[k].dtype == np.object:
                    # handle decompression
                    mini_batch_data[k] = np.asarray(
                        [mini_batch_data[k][i].decompress() for i in range(len(mini_batch_data[k]))])
                mini_batch_data[k] = torch.from_numpy(mini_batch_data[k]).to(runner.model.device)
        # todo: make this a with no log...
        runner.log.mode = runner.log.LM_MUTE
        mini_batch_func(mini_batch_data)
        runner.log.mode = runner.log.LM_DEFAULT
        # get small grad
        small_norms_sqr.append(utils.optimizer_grad_norm(optimizer) ** 2)
        if i == 0:
            big_grad = [x.clone() for x in utils.list_grad(optimizer)]
        else:
            for acc, p in zip(big_grad, utils.list_grad(optimizer)):
                acc += p

    optimizer.zero_grad(set_to_none=True)
    g_b_big_squared = float((utils.calc_norm(big_grad) / mini_batches) ** 2)
    g_b_small_squared = float(np.mean(small_norms_sqr))
    process_noise_scale(runner, g_b_small_squared, g_b_big_squared, label, verbose, b_big=b_big)


def get_value_head_accumulated_gradient_norms(
        runner: Runner,
        optimizer,
        prev_state,
        targets,
        required_head:int,
):
    """
    Calculate big and small gradient from given batch of data
    prev_state and targets should be in shuffled order.
    """

    B, K = targets.shape

    b_small = args.sns.b_small

    assert B % b_small == 0, "b_small must divide b_big"
    mini_batches = B // b_small

    small_norms_sqr = []
    big_grad = None

    for i in range(mini_batches):

        segment = slice(i*b_small, (i+1)*b_small)
        data = {"tvf_returns": targets[segment], "prev_state": prev_state[segment]}

        runner.log.mode = runner.log.LM_MUTE
        optimizer.zero_grad(set_to_none=True)
        runner.train_value_minibatch(data, single_value_head=-required_head)
        runner.log.mode = runner.log.LM_DEFAULT
        # get small grad
        small_norms_sqr.append(utils.optimizer_grad_norm(optimizer) ** 2)
        if i == 0:
            big_grad = [x.clone() for x in utils.list_grad(optimizer)]
        else:
            for acc, p in zip(big_grad, utils.list_grad(optimizer)):
                acc += p

    optimizer.zero_grad(set_to_none=True)

    # delete comment
    big_norm_sqr = (utils.calc_norm(big_grad)/mini_batches)**2

    return float(np.mean(small_norms_sqr)), float(big_norm_sqr)


def log_fake_accumulated_gradient_norms(runner:Runner, optimizer: torch.optim.Optimizer):

    required_heads = utils.even_sample_down(range(len(runner.tvf_horizons)), args.sns.max_heads)
    b_small = args.sns.b_small
    b_big = args.sns.b_big

    # get dims for this optimizer
    d = 0
    for group in optimizer.param_groups:
        for p in group['params']:
            if p.requires_grad:
                d += np.prod(p.data.shape)

    mini_batches = b_big // b_small

    for required_head in required_heads:

        small_norms_sqr = []
        big_grad = 0

        for i in range(mini_batches):
            # note: we do not use fake noise on final horizon, this is because I want to check if final head
            # and value noise estimate match, which they should as they measure the same thing.
            # note: we split half the noise as decreasing signal and the other half as increasing noise
            target_noise_level = runner.tvf_horizons[abs(required_head)] / 10
            if target_noise_level > 0:
                noise_level = math.sqrt(target_noise_level)
                signal_level = 1/math.sqrt(target_noise_level)
            else:
                noise_level = target_noise_level
                signal_level = 1
            grad = np.random.randn(d).astype(np.float32)
            norm2 = d ** 0.5 # a bit more fair than taking the true norm I guess
            # normalize so our noise vector is required length
            # the divide by b_small is because we would mean over these samples, so noise should be less
            renorm_factor = noise_level / norm2 / math.sqrt(b_small)
            grad *= renorm_factor
            grad[0] += signal_level # true signal is unit vector on first dim

            small_norms_sqr.append(np.linalg.norm(grad, ord=2) ** 2)
            if i == 0:
                big_grad = grad.copy()
            else:
                big_grad += grad

        g_small_sqr = float(np.mean(small_norms_sqr))
        g_big_sqr = (np.linalg.norm(big_grad, ord=2) / mini_batches) ** 2

        process_noise_scale(
            runner,
            g_small_sqr, g_big_sqr,
            label=f"fake_head_{required_head}",
            verbose=False
        )

def wants_noise_estimate(self, label:str):
    """
    Returns if given label wants a noise update on this step.
    """

    if not args.sns.enabled:
        return False
    if self.batch_counter % args.sns.period != args.sns.period-1:
        # only evaluate every so often.
        return False
    if label.lower() not in ast.literal_eval(args.sns.labels):
        return False
    return True


def log_accumulated_gradient_norms(rollout: Runner, batch_data):

    required_heads = utils.even_sample_down(range(len(rollout.tvf_horizons)), args.sns.max_heads)

    start_time = clock.time()
    for i, head_id in enumerate(required_heads):

        # select a different sample for each head (why not)
        prev_state = batch_data["prev_state"]
        targets = batch_data["tvf_returns"]
        if args.sns.b_big > rollout.N * rollout.A:
            raise ValueError(f"Can not take {args.sns.b_big} samples from rollout of size {rollout.N}x{rollout.A}")

        # we sample even when we need all examples, as it's important to shuffle the order
        sample = np.random.choice(range(rollout.N * rollout.A), args.sns.b_big, replace=False)
        prev_state = prev_state[sample]
        targets = targets[sample]

        g_small_sqr, g_big_sqr = get_value_head_accumulated_gradient_norms(
            rollout,
            optimizer=rollout.value_optimizer,
            prev_state=prev_state,
            targets=targets,
            required_head=head_id,
        )
        process_noise_scale(
            rollout,
            g_small_sqr, g_big_sqr, label=f"acc_head_{head_id}", verbose=False)
    s = clock.time() - start_time
    rollout.log.watch_mean("t_s_heads", s / args.sns.period)
