import os
import shlex
import subprocess
import numpy as np
import pandas as pd
from SentiCR.SentiCR.SentiCR import SentiCR


def clean_data(df):
    df = df.copy()

    # fill all rows with corresponding discussion link
    df[df['discussion_link'] == ""] = np.NaN
    df['discussion_link'] = df['discussion_link'].fillna(method='ffill')

    # keep only analyzed comments
    df = df[df['comment_ok'].notnull()]

    # identify contributors' comments
    df['contributor_comment'] = df['removed'].str.contains("contributor's comment")

    # save dataframe
    df_complete = df.copy()

    # remove all rows where change_ok is different than 'yes' or 'no'
    df = df[(df['change_ok'] == 'yes') | (df['change_ok'] == 'no')]

    # concatenate all answers to a discussion, as well as the last answer in the discussion, in a column
    answers = []
    last_answers = []
    last_answer_is_from_contributor = []
    for index, row in df.iterrows():
        row_answers = df_complete[(df_complete['discussion_link'] == row['discussion_link']) &
                                  (df_complete['filename'] == row['filename']) &
                                  (df_complete['commented_line'] == row['commented_line']) &
                                  (df_complete['comment'] != row['comment'])]

        if not row_answers.empty:
            last_answers.append(row_answers['comment'].iloc[-1])
            last_answer_is_from_contributor.append(row_answers['contributor_comment'].iloc[-1])
        else:
            last_answers.append(np.NaN)
            last_answer_is_from_contributor.append(np.NaN)
        answers.append(' '.join(row_answers['comment']))

    df['answers'] = answers
    df['last_answer'] = last_answers
    df['last_answer_is_from_contributor'] = last_answer_is_from_contributor

    # keep only one instance per discussion
    df = df.drop_duplicates(subset=['discussion_link', 'filename', 'method_signature', 'commented_line'], keep='first')

    # discard rows without answers
    df = df[df['answers'].str.len() > 0]

    return df


def merge_and_clean_data(df, df_complete):
    # fill all rows with corresponding discussion link
    df[df['discussion_link'] == ""] = np.NaN
    df['discussion_link'] = df['discussion_link'].fillna(method='ffill')

    # keep only analyzed comments
    df = df[df['comment_ok'].notnull()]

    # remove all rows where change_ok is different than 'yes' or 'no'
    df = df[(df['change_ok'] == 'yes') | (df['change_ok'] == 'no')]

    # add original commented line to dataframe
    for index, row in df.iterrows():
        match = df_complete[(df_complete['url'] == row['discussion_link']) &
                            (df_complete['filename'] == row['filename']) &
                            (df_complete['message'] == row['comment'])]

        if len(match) > 0:
            df.loc[index, 'commented_line'] = match['original_line'].iloc[0]

    # concatenate all answers to a discussion, as well as the last answer in the discussion, in a column
    answers = []
    last_answers = []
    last_answer_is_from_contributor = []
    for _, row in df.iterrows():
        row_answers = df_complete[(df_complete['url'] == row['discussion_link']) &
                                  (df_complete['filename'] == row['filename']) &
                                  (df_complete['original_line'] == row['commented_line']) &
                                  (df_complete['message'] != row['comment'])]

        # sort by creation date
        row_answers = row_answers.sort_values(by='created_at')

        if not row_answers.empty:
            last_answers.append(row_answers['message'].iloc[-1])
            last_answer_is_from_contributor.append(row_answers['owner_id'].iloc[-1] == row_answers['user_id'].iloc[-1])
        else:
            last_answers.append(np.NaN)
            last_answer_is_from_contributor.append(np.NaN)
        answers.append(' '.join(row_answers['message']))

    df['answers'] = answers
    df['last_answer'] = last_answers
    df['last_answer_is_from_contributor'] = last_answer_is_from_contributor

    # keep only one instance per discussion
    df = df.drop_duplicates(subset=['discussion_link', 'filename', 'method_signature', 'commented_line'], keep='first')

    # discard rows without answers
    df = df[df['answers'].str.len() > 0]

    return df


def extract_polarity(df, strategy, sa_tool='sentistrength'):
    # extract text to analyze depending on strategy
    df[strategy] = df[strategy].replace(r'[\n\t\r]', ' ', regex=True).replace(r'\"', '', regex=True)

    # run sentiment analysis
    if sa_tool == 'sentistrength':
        df.to_csv(f'{strategy}.tsv', sep='\t', columns=[strategy], index=True, header=False)
        sentiment_analysis_process = subprocess.Popen(shlex.split('java uk/ac/wlv/sentistrength/SentiStrength '
                                                                  f'sentidata {os.getcwd()}/SentiStrength-SE/ConfigFiles/ '
                                                                  f'input ../{strategy}.tsv '
                                                                  'annotateCol 2 overwrite '
                                                                  'trinary'),
                                                      cwd="SentiStrength-SE/")
        sentiment_analysis_process.communicate()

        # read results
        polarities = pd.read_csv(f'{strategy}.tsv', sep='\t',
                                 names=['original_index', 'text', 'polarity_sentistrength'])
        os.remove(f'{strategy}.tsv')
        return df.merge(polarities, left_index=True, right_on='original_index')
    else:
        sentiment_analyzer = SentiCR()
        df[f'polarity_senticr'] = df[strategy].apply(lambda x: sentiment_analyzer.get_sentiment_polarity(x)[0])
        return df


def build_oracle():
    df_all = clean_data(pd.read_csv('manual_analysis_all.csv'))
    df_filtered = merge_and_clean_data(pd.read_csv('manual_analysis_filtered.csv'),
                                       pd.read_csv('manual_analysis_filtered_complete.csv'))

    # add polarity to dataframe (separately because indexes overlap)
    df_all = extract_polarity(df_all, 'last_answer')
    df_filtered = extract_polarity(df_filtered, 'last_answer')
    df_all = extract_polarity(df_all, 'last_answer', 'senticr')
    df_filtered = extract_polarity(df_filtered, 'last_answer', 'senticr')

    # remove unnecessary columns
    df_all = df_all.drop(['original_index', 'removed', 'commented_file'], axis=1)
    df_filtered = df_filtered.drop(['can be identified as accepted'], axis=1)

    # combine the two dataframes
    df = pd.concat([df_all, df_filtered])

    df.to_csv('oracle.csv', index=False)

    return df


if __name__ == '__main__':
    build_oracle()
