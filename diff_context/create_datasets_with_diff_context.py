from replication.create_datasets import merge_data
from replication.create_datasets_split_by_time import split_data_by_time


def tag_code_and_comment_input(row):
    before_marked_context = row['before_marked_context'].replace(" END>", "<END>")

    return f"<code>{before_marked_context}</code>" \
           f"<technical_language>{row['comment_no_stopwords']}</technical_language>"


def add_file_context(row, column_name):
    return f"{row[column_name]}<SEP>{row['file_context']}"


def main():
    columns = ['before', 'before_context', 'before_marked_context', 'file_context',
               'comment', 'comment_no_stopwords', 'after', 'created_at']
    df = merge_data("../data/with_diff_context/", columns=columns)
    print("Total number of instances:", df.shape[0])

    # add diff context to the input
    df['before'] = df.apply(lambda row: add_file_context(row, 'before_context'), axis=1)
    df['code_and_comment'] = df.apply(tag_code_and_comment_input, axis=1)
    df['code_and_comment'] = df.apply(lambda row: add_file_context(row, 'code_and_comment'), axis=1)

    split_data_by_time(df, "../datasets/diff_context")


if __name__ == "__main__":
    main()
