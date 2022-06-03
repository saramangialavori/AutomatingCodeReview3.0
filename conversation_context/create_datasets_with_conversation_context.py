from replication.create_datasets import merge_data, tag_code_and_comment_input
from replication.create_datasets_split_by_time import split_data_by_time


def add_conversation_context(row, column_name):
    return f"{row[column_name]}<SEP>{row['conversation_context']}"


def main():
    columns = ['before', 'before_marked', 'conversation_context',
               'comment', 'comment_no_stopwords', 'after', 'created_at']
    df = merge_data("../data/with_conversation_context/", columns=columns)
    print("Total data size:", df.shape[0])

    # add conversation context
    df['before'] = df.apply(lambda row: add_conversation_context(row, 'before'), axis=1)
    df['code_and_comment'] = df.apply(tag_code_and_comment_input, axis=1)
    df['code_and_comment'] = df.apply(lambda row: add_conversation_context(row, 'code_and_comment'), axis=1)

    split_data_by_time(df, "../datasets/conversation_context")


if __name__ == '__main__':
    main()
