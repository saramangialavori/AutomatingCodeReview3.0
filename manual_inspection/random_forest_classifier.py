import getopt
import os
import sys

import pandas as pd
import weka.core.jvm as jvm
from weka.core.converters import Loader
from weka.core.stemmers import Stemmer
from weka.core.stopwords import Stopwords
from weka.core.tokenizers import Tokenizer
from weka.filters import Filter, StringToWordVector
from weka.classifiers import Classifier
from weka.classifiers import Evaluation
from weka.core.classes import Random
from build_oracle import clean_data, merge_and_clean_data, extract_polarity


def run_random_forest_classifier(df, strategy, balancing=True):  # strategy = 'answers' or 'last_answer'
    # save data to arff file
    with open(f'{strategy}.arff', 'w') as f:
        f.write('@relation ACR3\n')
        f.write(f'@attribute {strategy} string\n')
        f.write('@attribute length numeric\n')
        f.write('@attribute last_answer_contributor {True,False}\n')
        f.write('@attribute polarity numeric\n')
        f.write('@attribute change_ok {yes,no}\n')
        f.write('@data\n')

        for _, row in df.iterrows():
            f.write(f"\"{row[strategy]}\",{len(row[strategy])},{row['last_answer_is_from_contributor']},"
                    f"{row['polarity']},{row['change_ok']}\n")

    # start weka and load data
    jvm.start(packages='/Users/saramangialavori/wekafiles')  # TODO: modify with your path to weka files
    loader = Loader(classname="weka.core.converters.ArffLoader")
    data = loader.load_file(f'{strategy}.arff', class_index="last")

    # apply StringToWordVector filter
    s2wv = StringToWordVector(options=["-R", "first-last", "-N", "0", "-W", "1000", "-prune-rate", "1.0"])
    s2wv.stemmer = Stemmer(classname="weka.core.stemmers.NullStemmer")
    s2wv.stopwords = Stopwords(classname="weka.core.stopwords.Null", options=["-M", "1"])
    s2wv.tokenizer = Tokenizer(classname="weka.core.tokenizers.WordTokenizer",
                               options=["-delimiters", " \\r\\n\\t.,;:\\\'\\\"()?!"])
    s2wv.inputformat(data)
    data = s2wv.filter(data)

    if balancing:
        # reorder dataset
        reorder = Filter(classname="weka.filters.unsupervised.attribute.Reorder", options=["-R", "1-3,5-last,4"])
        reorder.inputformat(data)
        data = reorder.filter(data)

        # create training and test datasets
        remove_percentage = Filter(classname="weka.filters.unsupervised.instance.RemovePercentage",
                                   options=["-P", "20.0"])
        remove_percentage.inputformat(data)
        training = remove_percentage.filter(data)
        remove_percentage.options = ["-P", "20.0", "-V"]
        test = remove_percentage.filter(data)

        # balance dataset (training only)
        smote = Filter(classname="weka.filters.supervised.instance.SMOTE",
                       options=["-C", "0", "-K", "5", "-P", "100.0", "-S", "1"])
        smote.inputformat(training)
        training = smote.filter(training)

        data = training

    # build RandomForest classifier
    rf = Classifier(classname="weka.classifiers.trees.RandomForest",
                    options=["-P", "100", "-I", "100", "-num-slots", "1", "-K", "0",
                             "-M", "1.0", "-V", "0.001", "-S", "1"])
    rf.build_classifier(data)

    os.remove(f'{strategy}.arff')

    if balancing:
        return rf, data, test
    return rf, data, None


def evaluate_classifier(df, strategy, balancing):
    rf, data, test = run_random_forest_classifier(df, strategy, balancing)

    evaluation = Evaluation(data)
    if balancing:
        evaluation.test_model(rf, test)
    else:
        evaluation.crossvalidate_model(rf, data, 10, Random(1))
    print(evaluation.summary())
    print(evaluation.class_details())
    print(evaluation.matrix())

    jvm.stop()


def classify_instances(df, strategy):
    rf, data, _ = run_random_forest_classifier(df, strategy, False)

    predictions = []
    for index, instance in enumerate(data):
        prediction = rf.classify_instance(instance)
        predictions.append(instance.class_attribute.value(prediction))


def main(strategy, sa_tool='sentistrength', balancing=True):  # strategy = 'answers' or 'last_answer'
    df_all = clean_data(pd.read_csv('manual_analysis_all.csv'))
    df_filtered = merge_and_clean_data(pd.read_csv('manual_analysis_filtered.csv'),
                                       pd.read_csv('manual_analysis_filtered_complete.csv'))

    # add polarity to dataframe (separately because indexes overlap)
    df_all = extract_polarity(df_all, strategy, sa_tool)
    df_filtered = extract_polarity(df_filtered, strategy, sa_tool)

    # combine dataframes
    columns = ['discussion_link', 'comment', 'answers', 'last_answer',
               'last_answer_is_from_contributor', f'polarity_{sa_tool}', 'change_ok']
    df = pd.concat([df_all[columns], df_filtered[columns]])
    df = df.rename(columns={f'polarity_{sa_tool}': 'polarity'})

    evaluate_classifier(df, strategy, balancing)


if __name__ == '__main__':
    comments = 'last_answer'
    polarity = 'sentistrength'
    balanced = False
    try:
        opts, args = getopt.getopt(sys.argv[1:], 's:p:b', ['strategy=', 'polarity=', 'balanced'])
    except getopt.GetoptError:
        print('Expected usage: random_forest_classifier.py -s <strategy> -p <sentiment_analysis_tool> -b',
              file=sys.stderr)
        sys.exit(1)
    for opt, arg in opts:
        if opt in ['-s', '--strategy']:
            if arg in ['last_answer', 'answers']:
                comments = arg
            else:
                print('The strategy can either be "last_answer" or "answers", using "last_answer"', file=sys.stderr)
        if opt in ['-p', '--polarity']:
            if arg in ['sentistrength', 'senticr']:
                polarity = arg
            else:
                print('The polarity can either be "sentistrength" or "senticr", using "sentistrength"', file=sys.stderr)
        if opt in ['-b', '--balanced']:
            balanced = True
    main(comments, polarity, balanced)


