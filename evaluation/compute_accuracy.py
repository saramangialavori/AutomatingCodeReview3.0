import os
import re
import sys


def get_file_contents(folder, filename):
    with open(os.path.join(folder, filename)) as f:
        return f.readlines()


def main(folder_path):
    inputs = get_file_contents(folder_path, "inputs.txt")
    targets = get_file_contents(folder_path, "targets.txt")
    predictions = get_file_contents(folder_path, "predictions.txt")

    equal = 0
    for x, y, z in zip(predictions, targets, inputs):
        if re.sub(r'\s+', ' ', x) == re.sub(r'\s+', ' ', y):
            equal += 1

    print(f"Accuracy: {round(100 * equal / len(predictions), 3)}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Expected usage: compute_accuracy.py <folder_path>', file=sys.stderr)
        sys.exit(1)
    main(sys.argv[2])
