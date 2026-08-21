#!/usr/bin/env python3
# coding=utf-8

"""Calculate per-metric means and sample standard deviations for three runs."""

import statistics


def calculate_std(
    run1: list[float],
    run2: list[float],
    run3: list[float],
    metric_names: list[str] | None = None,
    precision: int = 4,
) -> list[dict[str, float | str]]:
    """Calculate each metric's mean and sample standard deviation."""
    if precision < 0:
        raise ValueError("precision 不能小于 0。")

    lengths = [len(run1), len(run2), len(run3)]
    if not run1:
        raise ValueError("每次运行的结果 list 不能为空。")
    if len(set(lengths)) != 1:
        raise ValueError(f"三个 list 长度必须相同，当前长度为 {lengths}。")

    metric_count = lengths[0]
    if metric_names is None:
        metric_names = [f"Metric {index + 1}" for index in range(metric_count)]
    elif len(metric_names) != metric_count:
        raise ValueError(
            f"metric_names 的长度必须是 {metric_count}，"
            f"当前为 {len(metric_names)}。"
        )

    results: list[dict[str, float | str]] = []
    header = (
        f"{'Metric':<20} {'Run 1':>10} {'Run 2':>10} "
        f"{'Run 3':>10} {'Mean':>10} {'Std':>10}"
    )
    print(header)
    print("-" * len(header))

    for name, value1, value2, value3 in zip(
        metric_names, run1, run2, run3
    ):
        values = [float(value1), float(value2), float(value3)]
        mean_value = statistics.fmean(values)
        std_value = statistics.stdev(values)
        results.append(
            {
                "metric": name,
                "mean": mean_value,
                "std": std_value,
            }
        )
        print(
            f"{name:<20} "
            f"{values[0]:>{10}.{precision}f} "
            f"{values[1]:>{10}.{precision}f} "
            f"{values[2]:>{10}.{precision}f} "
            f"{mean_value:>{10}.{precision}f} "
            f"{std_value:>{10}.{precision}f}"
        )

    print("\nMean +/- std:")
    for result in results:
        print(
            f"{result['metric']}: "
            f"{result['mean']:.{precision}f} +/- "
            f"{result['std']:.{precision}f}"
        )

    return results


if __name__ == "__main__":
    # 指标顺序必须与每个 run list 中的数值顺序一致。
    metric_names = [
        "BLEU-4",
        "METEOR",
        "ROUGE-L",
        "CIDEr",
        "BERTScore",
        "Precision",
        "Recall",
        "F1-Score",
    ]

    # 在这里分别填写三次完整实验的 8 个指标。
    #iu x-ray
    # medgemma baseline
    #run1 = [0.1071, 0.1882, 0.3745, 0.8786, 0.8028, 0.3445, 0.3875, 0.3477]
    #run2 = [0.1075, 0.1856, 0.3705, 0.8648, 0.8014, 0.3401, 0.3831, 0.3434]
    #run3 = [0.1084, 0.1872, 0.3729, 0.8695, 0.8024, 0.3415, 0.3823, 0.3441]
    # medgemma ours
    #run1 = [0.1319, 0.2246, 0.5064, 3.2226, 0.8391, 0.5075, 0.5531, 0.5067]
    #run2 = [0.1345, 0.2258, 0.5082, 3.2271, 0.8399, 0.5098, 0.5561, 0.5092]
    #run3 = [0.1306, 0.2240, 0.5049, 3.1817, 0.8384, 0.5072, 0.5538, 0.5069]
    # qwen3vl baseline
    #run1 = [0.0710, 0.1858, 0.3136, 0.6399, 0.7811, 0.2708, 0.3666, 0.2949]
    #run2 = [0.0706, 0.1857, 0.3156, 0.6296, 0.7815, 0.2718, 0.3705, 0.2968]
    #run3 = [0.0714, 0.1863, 0.3159, 0.6401, 0.7818, 0.2737, 0.3709, 0.2982]

    # qwen3vl ours

    #run1 = [0.2231, 0.2714, 0.5818, 3.7197, 0.8723, 0.5736, 0.6296, 0.5821]
    #run2 = [0.2247, 0.2723, 0.5830, 3.7338, 0.8728, 0.5768, 0.6326, 0.5847]
    #run3 = [0.2246, 0.2726, 0.5838, 3.7343, 0.8726, 0.5761, 0.6331, 0.5849]

    # mimic
    # medgemma baseline
    #run1 = [0.1118, 0.1974, 0.2598, 0.5702, 0.7807, 0.2466, 0.3238, 0.2565]
    #run2 = [0.1105, 0.1958, 0.2588, 0.5632, 0.7807, 0.2449, 0.3193, 0.2533]
    #run3 = [0.1099, 0.1964, 0.2574, 0.5657, 0.7808, 0.2478, 0.3248, 0.2568]
    # medgemma ours
    #run1 = [0.1060, 0.2244, 0.2884, 0.7723, 0.7852, 0.2516, 0.3893, 0.2798]
    #run2 = [0.1037, 0.2230, 0.2879, 0.7501, 0.7843, 0.2530, 0.3895, 0.2802]
    #run3 = [0.1028, 0.2220, 0.2863, 0.7500, 0.7842, 0.2526, 0.3878, 0.2797]
    # qwen3vl baseline
    #run1 = [0.0710, 0.1870, 0.2212, 0.3915, 0.7655, 0.1923, 0.3236, 0.2258]
    #run2 = [0.0701, 0.1880, 0.2188, 0.3887, 0.7656, 0.1914, 0.3251, 0.2256]
    #run3 = [0.0711, 0.1881, 0.2202, 0.3914, 0.7656, 0.1926, 0.3266, 0.2267]

    # qwen3vl ours

    #run1 = [0.0978, 0.2092, 0.2846, 0.8439, 0.7864, 0.2536, 0.3859, 0.2862]
    #run2 = [0.0978, 0.2088, 0.2835, 0.8308, 0.7859, 0.2515, 0.3839, 0.2837]
    #run3 = [0.0990, 0.2088, 0.2845, 0.8337, 0.7864, 0.2520, 0.3829, 0.2843]



    # jmid
    # medgemma baseline
    #run1 = [0.0672, 0.2192, 0.2531, 0.3241, 0.8731, 0.1027, 0.1935, 0.1195]
    #run2 = [0.0672, 0.2212, 0.2516, 0.3011, 0.8728, 0.0891, 0.1991, 0.1110]
    #run3 = [0.0681, 0.2208, 0.2523, 0.3155, 0.8730, 0.0851, 0.1821, 0.1055]
    # medgemma ours
    #run1 = [0.0977, 0.2479, 0.3284, 0.7729, 0.8848, 0.1659, 0.2327, 0.1763]
    #run2 = [0.0966, 0.2470, 0.3302, 0.7976, 0.8848, 0.1674, 0.2453, 0.1806]
    #run3 = [0.0975, 0.2472, 0.3294, 0.7892, 0.8847, 0.1681, 0.2299, 0.1747]
    # qwen3vl baseline
    #run1 = [0.0712, 0.2271, 0.2564, 0.2212, 0.8752, 0.0801, 0.1898, 0.1064]
    #run2 = [0.0701, 0.2265, 0.2551, 0.2119, 0.8751, 0.0873, 0.1992, 0.1136]
    #run3 = [0.0702, 0.2262, 0.2553, 0.2200, 0.8751, 0.0799, 0.1894, 0.1046]

    # qwen3vl ours

    run1 = [0.1187, 0.2667, 0.3412, 0.6509, 0.8873, 0.1728, 0.3070, 0.2019]
    run2 = [0.1191, 0.2656, 0.3412, 0.6702, 0.8872, 0.1657, 0.2951, 0.1952]
    run3 = [0.1192, 0.2657, 0.3413, 0.6579, 0.8873, 0.1667, 0.2928, 0.1931]

    calculate_std(run1, run2, run3, metric_names)
