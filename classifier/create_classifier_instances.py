import pandas as pd
from tqdm import tqdm
from replication.create_datasets import merge_data, create_tsv, clean
from replication.create_datasets_split_by_time import get_time_splits

good_as_is_instances = set()
all_before = set()


def add_classification_instances(df):
    global good_as_is_instances

    before = []
    classification = []
    original_instances = 0
    new_instances = 0
    invalid_rows = 0

    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        before.append(row['before'])
        classification.append(False)
        original_instances += 1

        if (str(row['merged']) == 'True') and (row['after'] not in good_as_is_instances):
            if row['after'] in all_before:
                continue
            new_instances += 1
            before.append(row['after'])
            classification.append(True)
            good_as_is_instances = good_as_is_instances.union({row['after']})

        elif str(row['merged']) != 'False' and str(row['merged']) != 'True':
            invalid_rows += 1

    df = pd.DataFrame({'before': before, 'classification': classification})

    df = df.drop_duplicates(subset=['before'])

    print(f"Original instances: {original_instances}")
    print(f"New instances: {new_instances}")
    print(f"Total instances: {df.shape[0]}")
    print(f"Invalid rows: {invalid_rows}")

    return df


def main():
    global all_before

    df = merge_data("../data/with_merging_info", with_merge_info=True)
    all_before = set(df['before'])
    print(f"Total instances: {df.shape[0]}")

    train, test, validation = get_time_splits(df)
    for df_split, name in [(train, "train"), (test, "test"), (validation, "val")]:
        df_split = df_split.copy()

        # add good-as-is instances
        print(f"Adding classification instances for {name}")
        df_split = add_classification_instances(df_split)
        df_split = df_split.drop_duplicates(subset=['before'])
        print(f"New {name} set size: {df_split.shape[0]}")

        # classifier
        create_tsv(df_split[['before', 'classification']], "../datasets/classifier/", name)


if __name__ == "__main__":
    main()
