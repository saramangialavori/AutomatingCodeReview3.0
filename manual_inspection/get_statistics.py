import nltk
import string
import pandas as pd
from collections import Counter
from itertools import chain


def clean_comment(comment):
    stopwords = ['just', 'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are',
                 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by',
                 'could', 'did', 'do', 'does', 'doing', 'down', 'during', 'each', 'few', 'for', 'from', 'further',
                 'had', 'has', 'have', 'having', 'he', 'he', 'her', 'here', 'hers', 'herself', 'him', 'himself', 'his',
                 'how', 'i', 'if', 'in', 'into', 'is', 'it', 'its', 'itself', 'me', 'more', 'most', 'my', 'myself',
                 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours',
                 'ourselves', 'out', 'over', 'own', 'same', 'she', 'should', 'so', 'some', 'such', 'than', 'that',
                 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'these', 'they', 'this', 'those',
                 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', 'we', 'were', 'what', 'when', 'where',
                 'which', 'while', 'who', 'whom', 'why', 'with', 'would', 'you', 'your', 'yours', 'yourself',
                 'yourselves']
    comment = comment.lower().translate(str.maketrans('', '', string.punctuation))
    return ' '.join([word for word in comment.split() if word not in stopwords])


def get_most_common_ngrams(df, change_ok, n):
    df['ngrams'] = df['last_answer'].map(lambda x: list(nltk.ngrams(x.split(" "), n)))

    ngrams = df[df['change_ok'] == change_ok]['ngrams'].tolist()
    ngrams = list(chain(*ngrams))

    ngram_counts = Counter(ngrams)
    return ngram_counts.most_common(10)


def main():
    df = pd.read_csv('oracle.csv')

    # Last answer term frequency
    print('########### Last answer term frequency ###########')
    df['last_answer'] = df['last_answer'].apply(lambda x: clean_comment(x))
    tf = df.assign(last_answer=df['last_answer'].str.split()).explode("last_answer") \
        .groupby("change_ok", sort=False)['last_answer'].value_counts()
    print(tf)
    tf.to_csv('last_answer_tf.csv')

    # Bigrams
    print('\n########### Bigrams ###########')
    print(get_most_common_ngrams(df, change_ok='yes', n=2))
    print(get_most_common_ngrams(df, change_ok='no', n=2))

    # Trigrams
    print('\n########### Trigrams ###########')
    print(get_most_common_ngrams(df, change_ok='yes', n=3))
    print(get_most_common_ngrams(df, change_ok='no', n=3))

    # Last answer length
    print('\n########### Last answer length ###########')
    df['last_answer_length'] = df['last_answer'].str.len()
    print(df.groupby('change_ok')['last_answer_length'].describe())

    # Last answer is from contributor
    print('\n########### Last answer is from contributor ###########')
    print(df.groupby('change_ok')['last_answer_is_from_contributor'].value_counts())

    # Polarity of last answer
    print('\n########### Polarity of last answer (SentiStrength-SE) ###########')
    print(df.groupby('change_ok')['polarity_sentistrength'].value_counts())
    print('\n########### Polarity of last answer (SentiCR) ###########')
    print(df.groupby('change_ok')['polarity_senticr'].value_counts())


if __name__ == '__main__':
    main()
