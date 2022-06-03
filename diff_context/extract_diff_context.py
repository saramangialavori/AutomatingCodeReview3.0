import os
import re
import shutil
import lizard
import javalang
import pandas as pd
from tqdm import tqdm
from difflib import Differ
from javalang.tree import *
from utils.github_requests import get_file_contents


def clean_method_lines(lines):
    block_comment = False

    for i in range(len(lines)):
        lines[i] = re.sub(r'\/\/.*', '', lines[i])
        if "/*" in lines[i]:
            if "*/" in lines[i]:
                lines[i] = re.sub(r'\/\*.*?\*\/', '', lines[i])
                lines[i] = lines[i].strip()
                continue
            lines[i] = re.sub(r'\/\*.*', '', lines[i])
            block_comment = True
        elif "*/" in lines[i]:
            lines[i] = re.sub(r'.*?\*\/', '', lines[i])
            block_comment = False
        elif block_comment:
            lines[i] = ""
        lines[i] = lines[i].strip()

    return list(filter(lambda x: len(x) > 0, lines))


def extract_methods(filename):
    names = []
    signatures = []
    parameters = []
    lines = []
    annotated_lines = []
    start_line = []

    code = [line for line in open(filename)]

    liz = lizard.analyze_file(filename)
    for method in liz.function_list:
        names.append(method.name.split('::')[-1])
        signatures.append(method.long_name)
        parameters.append(method.full_parameters)
        lines.append(clean_method_lines(list(map(lambda x: x.strip(), code[method.start_line - 1:method.end_line]))))
        annotated_lines.append([])
        start_line.append(method.start_line)

    return pd.DataFrame({'name': names, 'signature': signatures, 'parameters': parameters, 'lines': lines,
                         'annotated_lines': annotated_lines, 'start_line': start_line})


def split_method_lines(method_lines):
    if type(method_lines) == float:
        return []
    return [x.strip() for x in method_lines.split('<NEWLINE>')]


def compute_diff(before, after, keep_unchanged_lines=False):
    d = Differ()
    diff = []

    if keep_unchanged_lines:
        before = split_method_lines(before)
        after = split_method_lines(after)

    for line in list(d.compare(before, after)):
        if line.startswith('+ '):
            diff.append(("<ADD>", line[2:]))
        elif line.startswith('- '):
            diff.append(("<DEL>", line[2:]))
        elif line.startswith('  ') and keep_unchanged_lines:
            diff.append(("<UNC>", line[2:]))

    return list(filter(lambda x: len(x[1]) > 0, diff))


def add_method_invocation_context(method_name, parameters, liz_methods):
    context = ""

    matches = liz_methods[liz_methods['name'] == method_name]
    for _, method in matches.iterrows():
        if len(method['parameters']) == len(parameters):
            if method['context_added']:
                continue

            liz_methods.loc[liz_methods['signature'] == method['signature'], 'context_added'] = True

            if len(method['annotated_lines']) == 0:
                continue

            context += f"<MET>{method['signature']}"
            context += "".join(["".join(line) for line in method['annotated_lines']])

    return context


def add_file_context(method_name):
    file_context = ""

    # extract methods from before and while files
    before_methods = extract_methods('before.java')
    while_methods = extract_methods('while.java')

    # compute diff for methods in while file
    for idx, method in while_methods.iterrows():
        if method['signature'] in before_methods['signature'].values:
            before_method = before_methods[before_methods['signature'] == method['signature']].iloc[0]
            diff = compute_diff(before_method['lines'][1:], method['lines'][1:])
        else:
            # method was added
            diff = compute_diff([], method['lines'][1:])
        while_methods.at[idx, 'annotated_lines'] = diff

    for idx, method in before_methods.iterrows():
        if method['signature'] not in while_methods['signature'].values:
            # method was deleted
            diff = compute_diff(method['lines'][1:], [])
            before_methods.at[idx, 'annotated_lines'] = diff
            while_methods = pd.concat([while_methods, before_methods.iloc[idx].to_frame().T])

    liz_methods = while_methods
    liz_methods['context_added'] = False

    # find commented method and mark it as already added to context
    method_matches = liz_methods[liz_methods['signature'] == method_name]
    if len(method_matches) == 0:
        with open("errors.txt", "a") as f:
            f.write(f"Method {method_name} not found\n")
        return ""
    elif len(method_matches) > 1:
        with open("errors.txt", "a") as f:
            f.write(f"Multiple methods found for method {method_name}\n")
    liz_method = liz_methods[liz_methods['signature'] == method_name].iloc[0]
    liz_methods.loc[liz_methods.signature == method_name, 'context_added'] = True

    # get all methods and constructors in file using javalang
    try:
        tree = javalang.parse.parse(open("while.java").read())
    except (javalang.parser.JavaSyntaxError, javalang.tokenizer.LexerError) as error:
        with open("errors.txt", "a") as f:
            f.write(f"{error} for method {method_name}\n")
        return add_remaining_methods(file_context, liz_methods)

    jl_methods = [node for path, node in tree.filter(MethodDeclaration)]
    jl_methods += [node for path, node in tree.filter(ConstructorDeclaration)]

    # search for commented method in javalang methods
    method_matches = [x for x in jl_methods if x.position.line == liz_method['start_line']]
    if len(method_matches) == 0:
        with open("errors.txt", "a") as f:
            f.write(f"Method {method_name} not found in javalang tree\n")
        return add_remaining_methods(file_context, liz_methods)
    elif len(method_matches) > 1:
        with open("errors.txt", "a") as f:
            f.write(f"Multiple methods found in javalang tree for method {method_name}\n")
    jl_method = method_matches[0]

    # search for invocations to methods of same file
    method_invocations = [(node.member, node.arguments) for path, node in jl_method if type(node) == MethodInvocation]
    for member, arguments in method_invocations:
        if member in liz_methods['name'].values:
            file_context += add_method_invocation_context(member, arguments, liz_methods)

    # search for invocations to commented method from other methods in same file
    for method in jl_methods:
        for member, _ in [(node.member, node.arguments) for path, node in method if type(node) == MethodInvocation]:
            if member == liz_method['name']:
                file_context += add_method_invocation_context(method.name, method.parameters, liz_methods)

    # add remaining methods to context
    return add_remaining_methods(file_context, liz_methods)


def add_remaining_methods(file_context, liz_methods):
    for _, method in liz_methods[liz_methods['context_added'] == False].iterrows():
        if len(method['annotated_lines']) != 0:
            file_context += f"<MET>{method['signature']}"
            file_context += "".join(["".join(line) for line in method['annotated_lines']])
    return file_context


def only_contains_tags(line):
    line = line.replace('<START>', '')
    line = line.replace('<END>', '')
    line = line.replace('END>', '')
    line = line.strip()
    return len(line) == 0


def annotate_before_marked(before_diff, before_marked):
    before_marked = [x for x in split_method_lines(before_marked) if len(x) > 0]
    before_marked_diff = ""
    before_marked_idx = 0
    for annotation, line in before_diff:
        while before_marked_idx < len(before_marked) and only_contains_tags(before_marked[before_marked_idx]):
            before_marked_diff += before_marked[before_marked_idx]
            before_marked_idx += 1

        if annotation == "<DEL>":
            before_marked_diff += "<DEL>" + line
        else:
            if before_marked_idx < len(before_marked):
                before_marked_diff += annotation + before_marked[before_marked_idx]
                before_marked_idx += 1

    return before_marked_diff


def main():
    path_processed_data = '../data/processed'
    path_context_data = '../data/with_diff_context'
    if not os.path.exists(path_context_data):
        os.mkdir(path_context_data)

    files = [x for x in os.listdir(path_processed_data) if x not in os.listdir(path_context_data)]

    for file in tqdm(files, position=0, leave=True):
        filepath = os.path.join(path_processed_data, file)
        if os.stat(filepath).st_size == 0:
            shutil.copy(filepath, path_context_data)
            continue

        print(f'...Extracting context for {file}')
        with open('errors.txt', 'a') as f:
            f.write(file + '\n')
        df = pd.read_csv(filepath)

        if len(df) == 0:
            shutil.copy(filepath, path_context_data)
            continue

        before_context = []
        before_marked_context = []
        file_context = []
        for index, row in tqdm(df.iterrows(), total=len(df), position=1, leave=True):
            # annotate method lines with diff
            if row['commit_before'] == row['commit_while']:
                # if commit ids are the same, then the file is new
                before_diff = compute_diff('', row['before_lines'], True)
            else:
                before_diff = compute_diff(row['start_lines'], row['before_lines'], True)

            before_context.append("".join(["".join(line) for line in before_diff]))
            before_marked_context.append(annotate_before_marked(before_diff, row['before_marked_lines']))

            # add context from rest of file
            if row['commit_before'] == row['commit_while']:
                # if commit ids are the same, then the file is new
                file_context.append('')
            else:
                with open('before.java', 'w') as f:
                    f.write(get_file_contents(row['project'], row['commit_before'], row['filename']))

                with open('while.java', 'w') as f:
                    f.write(get_file_contents(row['project'], row['commit_while'], row['filename']))

                file_context.append(add_file_context(row['method_name']))

                os.remove('before.java')
                os.remove('while.java')

        df['before_context'] = before_context
        df['before_marked_context'] = before_marked_context
        df['file_context'] = file_context

        df.to_csv(os.path.join(path_context_data, file), index=False)

        os.remove('check.sh')
        os.remove('token.sh')


if __name__ == "__main__":
    main()
