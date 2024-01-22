from typing import List

import numpy as np
import scipy.integrate as integrate
import scipy.stats as st
import torch
import torch.nn.functional as F
from mpmath import mp, binomial

mp.dps = 15


def compute_log_nfa_anomaly_score(
        z: List[torch.Tensor],
        win_size: int = 5,
        binomial_probability_thr: float = 0.9,
        target_size: int = 256,
        high_precision: bool = False
):

    log_prob_l = [
        binomial_test(zi, win_size / (2 ** scale), binomial_probability_thr, high_precision)
        for scale, zi in enumerate(z)
    ]

    log_prob_l_up = [
        F.interpolate(lpl, size=(target_size, target_size), mode='bicubic', align_corners=True) for lpl in log_prob_l
    ]
    log_prob_l_up = torch.cat(log_prob_l_up, dim=1)

    log_prob = torch.sum(log_prob_l_up, dim=1, keepdim=True)

    log_number_of_tests = torch.log10(torch.sum(torch.tensor([zi.shape[-2] * zi.shape[-1] for zi in z])))
    log_nfa = log_number_of_tests + log_prob
    anomaly_score = -log_nfa

    return anomaly_score


def binomial_test(z: torch.Tensor, win, probability_thr: float, high_precision: bool = False):

    tau = st.chi2.ppf(probability_thr, 1)
    half_win = np.max([int(win // 2), 1])

    n_chann = z.shape[1]

    # Candidates
    z2 = F.pad(z ** 2, tuple(4 * [half_win]), 'reflect').detach().cpu()
    z2_unfold_h = z2.unfold(-2, 2 * half_win + 1, 1)
    z2_unfold_hw = z2_unfold_h.unfold(-2, 2 * half_win + 1, 1).numpy()
    observed_candidates_k = np.sum(z2_unfold_hw >= tau, axis=(-2, -1))

    # All volume together
    observed_candidates = np.sum(observed_candidates_k, axis=1, keepdims=True)
    x = observed_candidates / n_chann
    n = int((2 * half_win + 1) ** 2)

    # Low precision
    if not high_precision:
        log_prob = torch.tensor(st.binom.logsf(x, n, 1 - probability_thr) / np.log(10))
    # High precision - good and slow
    else:
        to_mp = np.frompyfunc(mp.mpf, 1, 1)
        mpn = mp.mpf(n)
        mpp = probability_thr
        binomial_density = lambda k: binomial(mpn, to_mp(k)) * ((1 - mpp) ** k) * (mpp ** (mpn - k))

        integral = lambda xx: integrate.quad(binomial_density, xx, n)[0]
        integral_array = np.vectorize(integral)
        prob = integral_array(x)
        log_prob = torch.tensor(np.log10(prob))

    return log_prob
