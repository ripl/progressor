import pdb
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def extract_event_data(file_path):
    # Load the event file
    event_acc = EventAccumulator(file_path)
    event_acc.Reload()
    # Extract scalar data
    tags = event_acc.Tags()['scalars']

    data = {tag: [] for tag in tags}
    for tag in tags:
        events = event_acc.Scalars(tag)
        for event in events:
            data[tag].append((event.wall_time, event.step, event.value))
    # Convert to pandas DataFrame
    dfs = []
    for tag, values in data.items():
        df = pd.DataFrame(values, columns=['wall_time', 'step', 'value'])
        df['tag'] = tag
        dfs.append(df)

    result_df = pd.concat(dfs, axis=0)
    return result_df


import csv,pdb
import glob, numpy as np
import pandas as pd
import matplotlib.pyplot as plt
folder_list = sorted(glob.glob('exp_local/*'))
print_var_list = ['train/episode_env_reward', 'train/batch_reward', 'train/episode_success_rate', 'train/actor_dormant_ratio']
total_data = {var:{} for var in print_var_list}
for folder_path in folder_list:
    data = None
    for event_file_list in sorted(glob.glob(folder_path + '/tb/*')):
        try:
            new_data = extract_event_data(event_file_list)
        except:
            continue
        if data is None:
            data = new_data
        else:
            data = data.append(new_data)
    value = data['value'].to_numpy()
    wall_time = data['wall_time'].to_numpy()
    step = data['step'].to_numpy()
    tag = data['tag'].to_numpy()
    label = folder_path.split('task=')[-1] + '-' + folder_path.split('reward_alpha=')[-1].split(',')[0]
    for var_name in print_var_list:
        if label not in total_data[var_name]:
            total_data[var_name][label] = []
        val = np.stack([step[tag == var_name], value[tag== var_name]], axis = -1)
        total_data[var_name][label].append(val)
for var in print_var_list:
    var_data = total_data[var]
    for key, val in var_data.items():
        min_len = min([data.shape[0] for data in val])
        data = np.stack([data[:min_len] for data in val])
        plt.plot(np.arange(min_len), data.mean(axis = 0)[:,1], label = key)
    plt.title(var)
    plt.legend()
    var = var.split('/')
    plt.savefig(f'{var[0]}_{var[1]}.png')
    plt.close()
pdb.set_trace()
