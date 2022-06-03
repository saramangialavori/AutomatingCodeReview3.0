import os


def get_stopwords():
    keywords = ['abstract', 'assert', 'boolean', 'break', 'byte', 'case', 'catch', 'char', 'class',
                'const', 'continue', 'default', 'do', 'double', 'else', 'enum', 'extends', 'final',
                'finally', 'float', 'for', 'if', 'goto', 'implements', 'import', 'instanceof', 'int',
                'interface', 'long', 'native', 'new', 'package', 'private', 'protected', 'public',
                'return', 'short', 'static', 'strictfp', 'super', 'switch', 'synchronized', 'this',
                'throw', 'throws', 'transient', 'try', 'void', 'volatile', 'while', 'true', 'false',
                'null', '+=', '&&', '-=', '!=', '++', '||', '<=', '>=', '<<=', '>>=', '...', '--',
                '/=', '>>>=', '<<<=', 'equals', 'inline', 'override']

    keywords = [k.upper() for k in keywords]

    working_dir = os.path.dirname(os.path.abspath(__file__))
    stopwords = [line.strip().upper() for line in open(os.path.join(working_dir, "stop-words-english.txt"))]
    stopwords.append('NIT')

    idioms = [line.strip().upper() for line in open(os.path.join(working_dir, "my_idioms_300.txt"))]

    for word in stopwords:
        if word in idioms or word in keywords:
            stopwords.pop(stopwords.index(word))

    return stopwords
