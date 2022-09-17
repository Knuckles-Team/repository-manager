#!/usr/bin/env python
# coding: utf-8

import subprocess
import os
import re
import sys
import getopt


class Git:
    def __init__(self):
        self.repository_directory = f"{os.getcwd()}"
        self.git_projects = []

    def git_action(self, command, directory=None):
        if directory is None:
            directory = self.repository_directory
        pipe = subprocess.Popen(command,
                                shell=True,
                                cwd=directory,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        (out, error) = pipe.communicate()
        result = f"{str(out, 'utf-8')}{str(error, 'utf-8')}"
        pipe.wait()
        return result

    def set_repository_directory(self, repository_directory):
        if os.path.exists(repository_directory):
            self.repository_directory = repository_directory.replace(os.sep, "/")
        else:
            print(f'Path specified does not exist: {repository_directory.replace(os.sep, "/")}')

    def set_git_projects(self, git_projects):
        self.git_projects = git_projects

    def append_git_project(self, git_project):
        self.git_projects.append(git_project)

    def clone_projects(self):
        for project in self.git_projects:
            print(self.git_action(f"git clone {project}"))

    def pull_projects(self, set_to_default_branch=False):
        for project_directory in os.listdir(self.repository_directory):
            print(f'Scanning: {self.repository_directory}/{project_directory}\n'
                  f'Pulling latest changes for {project_directory}\n'
                  f'{self.git_action(command="git pull", directory=f"{self.repository_directory}/{project_directory}")}')
            if set_to_default_branch:
                default_branch = self.git_action("git symbolic-ref refs/remotes/origin/HEAD", directory=f"{self.repository_directory}/{project_directory}")
                default_branch = re.sub("refs/remotes/origin/", "", default_branch).strip()
                print(f"Checking out default branch ", self.git_action(f'git checkout "{default_branch}"', directory=f"{self.repository_directory}/{project_directory}"))


def usage():
    print(f"Usage: \n"
          f"-h | --help           [ See usage for script ]\n"
          f"-b | --default-branch [ Checkout default branch ]\n"
          f"-c | --clone          [ Clone projects specified  ]\n"
          f"-d | --directory      [ Directory to clone/pull projects ]\n"
          f"-f | --file           [ File with repository links   ]\n"          
          f"-p | --pull           [ Pull projects in parent directory ]\n"          
          f"-r | --repositories   [ Comma separated Git URLs ]\n"
          f"\n"
          f"repository-manager --clone --pull --directory '/home/user/Downloads' \\\n"
          f"--file '/home/user/Downloads/repositories.txt' \\\n"
          f"--repositories 'https://github.com/Knucklessg1/media-downloader,https://github.com/Knucklessg1/genius-bot'")


def repository_manager(argv):
    gitlab = Git()
    projects = []
    default_branch_flag = False
    clone_flag = False
    pull_flag = False
    directory = os.curdir
    file = None
    repositories = None
    try:
        opts, args = getopt.getopt(argv, "hbcpd:f:r:", ["help", "default-branch", "clone", "pull", "directory=", "file=", "repositories="])
    except getopt.GetoptError:
        usage()
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            usage()
            sys.exit()
        elif opt in ("-b", "--b"):
            default_branch_flag = True
        elif opt in ("-c", "--clone"):
            clone_flag = True
        elif opt in ("-p", "--pull"):
            pull_flag = True
        elif opt in ("-d", "--directory"):
            directory = arg
        elif opt in ("-f", "--file"):
            file = arg
        elif opt in ("-r", "--repositories"):
            repositories = arg.replace(" ", "")
            repositories = repositories.split(",")

    # Verify directory to clone/pull exists
    if os.path.exists(directory):
        gitlab.set_repository_directory(directory)
    else:
        print(f"Directory not found: {directory}")
        usage()
        sys.exit(2)

    # Verify file with repositories exists
    if os.path.exists(file):
        file_repositories = open(file, 'r')
        for repository in file_repositories:
            projects.append(repository)
    else:
        print(f"File not found: {file}")
        usage()
        sys.exit(2)

    if repositories:
        for repository in repositories:
            projects.append(repository)

    projects = list(dict.fromkeys(projects))

    gitlab.set_git_projects(projects)

    if clone_flag:
        gitlab.clone_projects()
    if pull_flag:
        gitlab.pull_projects(set_to_default_branch=default_branch_flag)


def main():
    if len(sys.argv) < 2:
        usage()
        sys.exit(2)
    repository_manager(sys.argv[1:])


if __name__ == "__main__":
    if len(sys.argv) < 2:
        usage()
        sys.exit(2)
    repository_manager(sys.argv[1:])
