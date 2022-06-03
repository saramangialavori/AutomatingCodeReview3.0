import csv
import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split


def merge_data(data_folder, columns=None, with_merge_info=False):
    def clean_and_drop_duplicates(df):
        df['comment'].fillna("null", inplace=True)
        df['comment_no_stopwords'].fillna("null", inplace=True)
        df.fillna("", inplace=True)
        df.drop_duplicates(subset=['before'], inplace=True)
        return df

    merge_file = "merged.csv"
    if os.path.exists(merge_file):
        return clean_and_drop_duplicates(pd.read_csv(merge_file))

    if columns is None:
        columns = ['project', 'pull_num', 'commit_before', 'commit_while', 'filename', 'method_name', 'comment',
                   'created_at', 'start', 'before', 'before_marked', 'after', 'comment_no_stopwords', 'start_lines',
                   'before_lines', 'before_marked_lines']
    if with_merge_info:
        columns += ['merged']

    merged_df = pd.DataFrame(columns=columns)

    files = [file for file in os.listdir(data_folder)]
    for file in files:
        filepath = os.path.join(data_folder, file)

        # skip if file is empty
        if os.path.getsize(filepath) == 0:
            continue

        file_df = pd.read_csv(filepath)
        try:
            merged_df = pd.concat([merged_df, file_df[columns]])
        except KeyError:
            # columns are invalid e.g. if the file was not processed
            # as it only contains the header
            continue

    for column in merged_df.columns:
        merged_df[column] = merged_df[column].apply(clean)

    merged_df.to_csv(merge_file, index=False)

    # read from file to handle nan values which where converted to string
    return clean_and_drop_duplicates(pd.read_csv(merge_file))


def clean(string):
    string = str(string).strip()
    string = string.replace('\n', ' ')
    string = string.replace('\\n', ' ')
    string = string.replace('\t', ' ')
    string = string.replace('\\t', ' ')
    string = re.sub(r'\s+', ' ', string)
    return string


def tag_code_and_comment_input(df):
    return f"<code>{df['before_marked']}</code><technical_language>{df['comment_no_stopwords']}</technical_language>"


def create_tsv(df, folder, name):
    if not os.path.exists(folder):
        os.mkdir(folder)
    df.to_csv(os.path.join(folder, name + ".tsv"), sep='\t', index=False, header=False, quoting=csv.QUOTE_NONE)


def split_data(df, folder='.', splits=None):
    df = df.copy()

    if splits is None:
        train, test_and_validation = train_test_split(df, test_size=0.2, random_state=1)
        test, validation = train_test_split(test_and_validation, test_size=0.5, random_state=1)
    else:
        train, test, validation = splits

    print("Train set size:", train.shape[0])
    print("Test set size:", test.shape[0])
    print("Validation set size:", validation.shape[0])

    if not os.path.exists(folder):
        os.mkdir(folder)

    for df_split, name in [(train, "train"), (test, "test"), (validation, "val")]:
        df_split = df_split.copy()

        # code-to-code
        create_tsv(df_split[['before', 'after']], os.path.join(folder, "code-to-code"), name)

        # code-to-comment
        create_tsv(df_split[['before', 'comment']], os.path.join(folder, "code-to-comment"), name)

        # code&comment-to-code
        create_tsv(df_split[['code_and_comment', 'after']], os.path.join(folder, "code&comment-to-code"), name)


def main():
    df = merge_data("../data/processed")
    print("Total number of instances:", df.shape[0])

    # prepare code&comment-to-code input
    df['code_and_comment'] = df.apply(tag_code_and_comment_input, axis=1)

    split_data(df, folder='../datasets/replication')


if __name__ == '__main__':
    main()
