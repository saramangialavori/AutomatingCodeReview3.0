import getopt
import os.path
import sys
import pandas as pd
from build_oracle import build_oracle, clean_data, merge_and_clean_data, extract_polarity
from get_statistics import clean_comment


def evaluate_heuristics(df):
    if not os.path.exists('oracle.csv'):
        build_oracle()
    oracle = pd.read_csv('oracle.csv')

    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for index, row in oracle.iterrows():
        match = df[(df['discussion_link'] == row['discussion_link']) &
                   (df['filename'] == row['filename']) &
                   (df['method_signature'] == row['method_signature']) &
                   (df['comment'] == row['comment']) &
                   (df['commented_line'] == row['commented_line'])]

        if len(match) > 0:
            if row['change_ok'] == 'yes':
                if match['filtered_out'].iloc[0]:  # equivalent to change_ok=no
                    fn += 1
                else:
                    tp += 1
            else:
                if match['filtered_out'].iloc[0]:
                    tn += 1
                else:
                    fp += 1
        else:
            print("No match found for:", row)

    print(f"Correctly classified instances:\t\t{tp+tn}\t{(tp+tn)/(tp+tn+fp+fn)*100}%")
    print(f"Incorrectly classified instances:\t{fp+fn}\t{(fp+fn)/(tp+tn+fp+fn)*100}%")
    print(f"Total number of instances:\t\t\t{tp+tn+fp+fn}")

    print("Precision:", tp / (tp + fp))
    print("Recall:", tp / (tp + fn))
    print("Accuracy:", (tp + tn) / (tp + tn + fp + fn))
    print("F1:", 2 * tp / (2 * tp + fp + fn))

    print("yes\tno\t<-- classified as")
    print(f"{tp}\t{fn}\t|\tyes")
    print(f"{fp}\t{tn}\t|\tno")


def filter_out_long_answers(df):
    """Filter out suggestions whose last answer is longer than 70 words."""

    df['last_answer_length'] = df['last_answer'].apply(lambda x: len(x.split()))
    df.loc[df['last_answer_length'] >= 70, 'filtered_out'] = True

    return df


def last_answer_keywords_heuristics(df):
    """Filter out or keep suggestions whose last answer contains certain keywords."""
    keywords_keep = ["done", "checked", "fixed", "changed", "correct", "removed", "resolved", "updated", "thanks",
                     "thank much", "good idea", "will fix", "good catch", "will cleanup"]
    keywords_remove = ["create confusion", "cant fix"]

    for index, row in df.iterrows():
        cleaned_answer = clean_comment(row['last_answer'])
        for keyword in keywords_keep:
            if keyword in cleaned_answer:
                df.at[index, 'filtered_out'] = False
        for keyword in keywords_remove:
            if keyword in cleaned_answer:
                df.at[index, 'filtered_out'] = True

    return df


def filter_out_negative_answers(df, sa_tool):
    """Filter out suggestions whose last answer has negative polarity."""
    df.loc[df[f'polarity_{sa_tool}'] < 0, 'filtered_out'] = True

    return df


def run_heuristics(sa_tool=None):  # sa_tool = 'sentistrength' or 'senticr'
    df_all = clean_data(pd.read_csv('manual_analysis_all.csv'))
    df_filtered = merge_and_clean_data(pd.read_csv('manual_analysis_filtered.csv'),
                                       pd.read_csv('manual_analysis_filtered_complete.csv'))

    if sa_tool is not None:
        # add polarity to dataframe (separately because indexes overlap)
        df_all = extract_polarity(df_all, 'last_answer', sa_tool)
        df_filtered = extract_polarity(df_filtered, 'last_answer', sa_tool)

    # combine dataframes
    columns = ['discussion_link', 'comment', 'filename', 'method_signature', 'commented_line',
               'answers', 'last_answer', 'last_answer_is_from_contributor', 'change_ok']
    if sa_tool is not None:
        columns.append(f'polarity_{sa_tool}')
    df = pd.concat([df_all[columns], df_filtered[columns]])

    # initialize filtering
    df['filtered_out'] = False  # this is equivalent to change_ok = yes

    # apply heuristics
    df = filter_out_long_answers(df)
    if sa_tool is not None:
        df = filter_out_negative_answers(df, sa_tool)
    df = last_answer_keywords_heuristics(df)

    return df


if __name__ == '__main__':
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'p:', ['polarity='])
    except getopt.GetoptError:
        print('Expected usage: heuristics.py -p <sentiment_analysis_tool>', file=sys.stderr)
        sys.exit(1)
    for opt, arg in opts:
        if opt in ['-p', '--polarity']:
            if arg in ['sentistrength', 'senticr']:
                print('Executing heuristic with polarity', arg)
                evaluate_heuristics(run_heuristics(sa_tool=arg))
            else:
                print('Expected either "sentistrength" or "senticr", executing without polarity', file=sys.stderr)
                evaluate_heuristics(run_heuristics())
    if len(opts) == 0:
        print('Executing heuristic without polarity')
        evaluate_heuristics(run_heuristics())
