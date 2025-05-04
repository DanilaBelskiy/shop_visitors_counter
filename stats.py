import json
import os

from app_config import stats_folder


def save_statistic(stat_dict, fps, filename):

    amount_by_seconds = {}
    max_amount = 0
    min_amount = float('inf')

    for second in range(len(stat_dict.keys()) // fps):

        second_amount = 0

        for i in range(second * fps, (second+1) * fps):
            second_amount += stat_dict[i]

            if stat_dict[i] > max_amount:
                max_amount = stat_dict[i]

            if stat_dict[i] < min_amount:
                min_amount = stat_dict[i]

        amount_by_seconds[second] = second_amount // fps

    data = {
        "max_customers": max_amount,
        "min_customers": min_amount,
        "amount_by_seconds": amount_by_seconds,
    }

    with open(os.path.join(stats_folder, f"{filename.split('.')[0]}.json"), 'w') as f:
        json.dump(data, f)
