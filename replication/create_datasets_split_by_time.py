import pandas as pd
from replication.create_datasets import merge_data, tag_code_and_comment_input, split_data
from sklearn.model_selection import train_test_split


def get_time_splits(df):
    df['created_at'] = pd.to_datetime(df['created_at'])
    df.sort_values(by='created_at', inplace=True)

    train_size = round(df.shape[0] * 0.8)

    train = df.iloc[:train_size]
    test, validation = train_test_split(df.iloc[train_size:], test_size=0.5, random_state=1)
    test.sort_values(by='created_at', inplace=True)
    validation.sort_values(by='created_at', inplace=True)

    return train, test, validation


def split_data_by_time(df, folder):
    split_data(df, splits=get_time_splits(df), folder=folder)


def main():
    df = merge_data("../data/processed")
    print("Total number of instances:", df.shape[0])

    # prepare code&comment-to-code input
    df['code_and_comment'] = df.apply(tag_code_and_comment_input, axis=1)

    split_data_by_time(df, "../datasets/replication_by_time")


if __name__ == '__main__':
    main()
