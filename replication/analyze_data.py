import os
import csv
import shutil
import pandas as pd
from tqdm import tqdm
from Analyzer import Analyzer
from Cleaner import Cleaner
from transformers import T5Tokenizer
from utils.stopwords import get_stopwords


def analyze_data(df):
    df.reset_index(inplace=True)

    analyzer = Analyzer(df, 'GitHub')  # GitHub or Gerrit
    analyzer.remove_contributor_comments()
    analyzer.remove_nan_data()

    return analyzer.remove_invalid_data()


def clean_data(df_to_analyze, previous_predictions, t5_tokenizer, stopwords):
    cleaner = Cleaner(df_to_analyze, t5_tokenizer, stopwords, previous_predictions)
    cleaner.clean_df()
    cleaner.remove_multiple_method_comments()

    return cleaner.get_df()


def main():
    path_data_folder = '../data'
    path_processed_data = os.path.join(path_data_folder, 'processed')  # processed data folder path
    if not os.path.exists(path_processed_data):
        os.mkdir(path_processed_data)

    files = [file for file in os.listdir(path_data_folder)
             if file not in os.listdir(path_processed_data)
             and (file.endswith('.csv') and not file.endswith('_summary.csv') and not file.endswith('_stats.csv'))]

    t5_tokenizer = T5Tokenizer.from_pretrained("../tokenizer/TokenizerModel.model")
    stopwords = get_stopwords()

    for file in tqdm(files):
        print(f'Analyzing file: {file}')

        # check if file is empty
        if os.path.getsize(os.path.join(path_data_folder, file)) == 0:
            print(f'...File {file} is empty, skipping...')
            shutil.copy(os.path.join(path_data_folder, file), path_processed_data)
            continue

        # read tsv of previous language predictions
        open('english_real_predictions.tsv', 'a').close()
        df_previous_predictions = pd.read_csv('english_real_predictions.tsv', sep='\t',
                                              names=['comment', 'confidence', 'lang'],
                                              quoting=csv.QUOTE_NONE)

        # analyze and clean data
        df = pd.read_csv(os.path.join(path_data_folder, file), index_col=0)
        df = analyze_data(df)
        df = clean_data(df, df_previous_predictions, t5_tokenizer, stopwords)

        # when a comment like "why null?" is processed, only null is left, and pandas interprets it as a NaN
        df = df.fillna('null')

        # discard all the remaining duplicates
        df = df.drop_duplicates(subset=["before"])

        # save data to processed folder
        df.to_csv(os.path.join(path_processed_data, file))

        # remove unnecessary files
        os.remove('start.java')
        os.remove('before.java')
        os.remove('after.java')


if __name__ == '__main__':
    main()
