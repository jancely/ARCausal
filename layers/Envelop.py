import torch

def extract_envelop(signals):    # [bs, seq_len, nvars]
    """
    Extract Series upper or lower envelop
    Parameters：
    - signal: Input univariate or multivariate  (torch.Tensor)
    - window_size: smooth filtering window size
    - threshold: threshold of maximum or minimum

    Return：
    - upper_envelope
    - lower_envelope
    """

    # Avg-pooling smoothing
    # signal = signal.permute(1, 0)

    # for i in range(len(signals)):    # Batch, Length, Variates
        # smoothed_signal = F.avg_pool1d(signal[i].unsqueeze(0), kernel_size=window_size, padding=window_size // 2, stride=1).squeeze(0)
        # signal = signals[i]

        # Locating maximum and minimum
    is_max = torch.zeros_like(signals)
    is_min = torch.zeros_like(signals)

    is_max[:, 1:-1, :] = (signals[:, 1:-1, :] > signals[:, :-2, :]) & (
                signals[:, 1:-1, :] > signals[:, 2:, :])
    is_min[:, 1:-1, :] = (signals[:, 1:-1, :] < signals[:, :-2, :]) & (
                signals[:, 1:-1, :] < signals[:, 2:, :])

    # Extracting maximum or minimum of series
    upper_envelope = torch.zeros_like(signals)
    lower_envelope = torch.zeros_like(signals)

    upper_envelope[is_max.bool()] = signals[is_max.bool()]
    lower_envelope[is_min.bool()] = signals[is_min.bool()]

    # Filtering miner points of envelop
    # upper_envelope[upper_envelope < threshold] = float('nan')
    # lower_envelope[lower_envelope > threshold] = float('nan')

    # for i in range(len(smoothed_signal)):
    upper_index = torch.nonzero(upper_envelope)
    lower_index = torch.nonzero(lower_envelope)

    return upper_envelope, lower_envelope, upper_index, lower_index


def filter(signals):
    signals_fft = torch.fft.fft(signals, dim=1)
    am = torch.abs(signals_fft)
    _, top_indices = torch.topk(am, k=24, dim=1)
    mask1 = torch.zeros_like(signals_fft)

    for i in range(signals_fft.shape[0]):
        mask1[i, top_indices[i], :] = 1

    signals_fft = signals_fft * mask1

    signals1 = torch.fft.ifft(signals_fft, dim=1).real
    signals2 = signals - signals1

    return signals1, signals2