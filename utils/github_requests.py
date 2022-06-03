import json
import time
import subprocess

# TODO: update with your GitHub username and token
username = "username"
token = "token"


def check_remaining_requests_github():
    with open('./check.sh', 'w') as f:
        f.write(f'#!/bin/sh\ncurl -H "Authorization: token {token}" -X GET https://api.github.com/rate_limit')
    output = subprocess.run(['/bin/sh', './check.sh'], capture_output=True).stdout
    remaining = json.loads(output)['rate']['remaining']
    if int(remaining) > 0:
        return True
    return False


def request_link(link):
    with open('./token.sh', 'w') as f:
        f.write(f'#!/bin/sh\ncurl -u {username}:{token} {link}')
    while not check_remaining_requests_github():
        time.sleep(5)
    return subprocess.run(['/bin/sh', './token.sh'], capture_output=True).stdout


def get_file_contents(project, commit, filename):
    return request_link(f"https://raw.githubusercontent.com/{project}/{commit}/{filename}").decode('utf-8', 'replace')


def get_merge_commit_id(project, pull_number):
    output = request_link(f"https://api.github.com/repos/{project}/pulls/{pull_number}")
    return json.loads(output)['merge_commit_sha']
