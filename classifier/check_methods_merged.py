import os
import shutil
import sys

import pandas as pd
from tqdm import tqdm
from replication.create_datasets import clean
from utils.github_requests import get_file_contents, get_merge_commit_id


def main():
    path_data_folder = '../data'
    path_processed_data = os.path.join(path_data_folder, 'processed')
    path_merged_info = os.path.join(path_data_folder, 'with_merging_info')
    if not os.path.exists(path_merged_info):
        os.mkdir(path_merged_info)

    files = [file for file in os.listdir(path_processed_data) if file not in os.listdir(path_merged_info)]

    warnings = 0
    no_merging = 0
    for file in tqdm(files):
        print(f'Analyzing file: {file}')

        # skip if file is empty
        filepath = os.path.join(path_processed_data, file)
        if os.stat(filepath).st_size == 0:
            shutil.copy(filepath, path_merged_info)
            continue

        merged_bool = []
        df = pd.read_csv(filepath)
        n = 0
        project = ''
        filename = ''
        merged_code = ''
        no_merging_flag = False
        for i in tqdm(range(len(df))):
            current_n = df.iloc[i]['pull_num']
            current_project = df.iloc[i]['project']
            current_filename = df.iloc[i]['filename']
            # check if same PR and same file of previous instance
            if current_n == n and current_project == project and current_filename == filename and no_merging_flag:
                merged_bool.append(False)
                continue
            elif not (current_n == n and current_project == project and current_filename == filename):
                no_merging_flag = False
                n = current_n
                project = current_project
                filename = current_filename
                try:
                    # retrieve merged commit id
                    merge_commit_id = get_merge_commit_id(project, n)
                except Exception as e:
                    print(e, file=sys.stderr)
                    no_merging_flag = True
                    no_merging += 1
                    merged_bool.append(False)
                    continue
                if not merge_commit_id:
                    no_merging += 1
                    merged_bool.append(False)
                    continue
                try:
                    merged_code = get_file_contents(project, merge_commit_id, filename)
                except Exception as e:
                    print(e, file=sys.stderr)
                    no_merging_flag = True
                    merged_bool.append(False)
                    warnings += 1
                    continue
            # check if merged file contains modified method
            code_after = df.iloc[i]['after']
            if code_after in clean(merged_code):
                merged_bool.append(True)
            else:
                merged_bool.append(False)
        df['merged'] = merged_bool
        df.to_csv(os.path.join(path_merged_info, file))

        os.remove('check.sh')
        os.remove('token.sh')

    print('no merging:', no_merging)
    print('warnings:', warnings)


if __name__ == '__main__':
    main()
