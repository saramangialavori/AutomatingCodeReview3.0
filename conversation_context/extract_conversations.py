import csv
import os
import shutil
import pandas as pd
from tqdm import tqdm
from transformers import T5Tokenizer
from replication.Analyzer import Analyzer
from replication.Cleaner import Cleaner
from utils.stopwords import get_stopwords


def get_cleaner_instance():
    previous_predictions = pd.read_csv('../replication/english_real_predictions.tsv', sep='\t',
                                       names=['comment', 'confidence', 'lang'],
                                       quoting=csv.QUOTE_NONE)

    t5_tokenizer = T5Tokenizer.from_pretrained("../tokenizer/TokenizerModel.model")
    cleaner = Cleaner(None, t5_tokenizer, get_stopwords(), previous_predictions)

    return cleaner


def clean_comment(comment, ref, cleaner):
    if Analyzer.check_comment_to_comment(ref[0], ref[1]):
        return None

    comment = Cleaner.replace_links(comment)[0]
    comment = Cleaner.remove_emojis(comment)
    comment = Cleaner.clean_string(comment)
    comment = Cleaner.replace_symbols(comment)

    if Cleaner.is_non_latin(comment):
        return None

    if not cleaner.is_english(comment):
        return None

    return comment


def add_user_tag(comment, _, is_owner, cleaner):
    comment = cleaner.remove_stopwords(comment)
    if is_owner:
        return f"<AUT>{comment}"
    else:
        return f"<REV>{comment}"


def main():
    path_processed_data = '../data/processed'
    path_context_data = '../data/with_conversation_context'
    if not os.path.exists(path_context_data):
        os.mkdir(path_context_data)

    files = [x for x in os.listdir(path_processed_data) if x not in os.listdir(path_context_data)]

    cleaner = get_cleaner_instance()

    for file in tqdm(files, leave=True, position=0):
        print("...Processing file:", file)
        filepath = os.path.join(path_processed_data, file)
        if os.stat(filepath).st_size == 0:
            shutil.copy(filepath, path_context_data)
            continue

        df_processed = pd.read_csv(filepath)
        df_processed['created_at'] = pd.to_datetime(df_processed['created_at'])

        df_original = pd.read_csv(os.path.join('../data', file))
        df_original['file_content_before'] = df_original['file_content_before'].fillna('')
        df_original['created_at'] = pd.to_datetime(df_original['created_at'])

        conversation_context = []
        for index, row in tqdm(df_processed.iterrows(), total=len(df_processed), leave=True, position=1):
            matches = df_original[(df_original['pull_number'] == row['pull_num']) &
                                  (df_original['pull_id'] == row['pull_id']) &
                                  (df_original['filename'] == row['filename'])]

            valid_comments = []
            for _, match in matches.iterrows():
                # check method signature matches
                _, _, ref = Analyzer.get_info_github(match)
                Analyzer.save_temp_code(match, 'GitHub')
                method_found = Analyzer.search_before_method(ref)
                if (len(method_found) == 1) and (method_found[0].long_name == row['method_name']):
                    comment = clean_comment(match['message'], ref, cleaner)
                    if comment is not None:
                        valid_comments.append((comment, match['created_at'], match['user_id'] == match['owner_id']))

            # discard all comments after original comment
            valid_comments = [x for x in valid_comments if x[1] < row['created_at']]
            valid_comments = sorted(valid_comments, key=lambda x: x[1], reverse=True)

            conversation_context.append(''.join([add_user_tag(*x, cleaner) for x in valid_comments]))

        df_processed['conversation_context'] = conversation_context
        df_processed.to_csv(os.path.join(path_context_data, file), index=False)

        os.remove('start.java')
        os.remove('before.java')
        os.remove('after.java')


if __name__ == '__main__':
    main()
