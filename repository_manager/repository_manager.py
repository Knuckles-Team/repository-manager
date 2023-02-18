#!/usr/bin/env python
# coding: utf-8

import subprocess
import os
import re
import sys
import getopt
from multiprocessing import Pool


class Git:
    def __init__(self):
        self.repository_directory = f"{os.getcwd()}"
        self.git_projects = []
        self.set_to_default_branch = False
        self.threads = os.cpu_count()

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

    def set_default_branch(self, set_to_default_branch):
        self.set_to_default_branch = set_to_default_branch

    def set_threads(self, threads):
        try:
            threads = int(threads)
            if threads > 0 or threads < os.cpu_count():
                self.threads = threads
            else:
                print(f"Did not recognize {threads} as a valid value, defaulting to CPU Count: {os.cpu_count()}")
                self.threads = os.cpu_count()
        except Exception as e:
            print(f"Did not recognize {threads} as a valid value, defaulting to CPU Count: {os.cpu_count()}\nError: {e}")
            self.threads = os.cpu_count()

    def append_git_project(self, git_project):
        self.git_projects.append(git_project)

    def clone_projects_in_parallel(self):
        pool = Pool(processes=self.threads)
        pool.map(self.clone_project, self.git_projects)

    def clone_project(self, git_project):
        print(self.git_action(f"git clone {git_project}"))

    def pull_projects_in_parallel(self):
        pool = Pool(processes=self.threads)
        pool.map(self.pull_project, os.listdir(self.repository_directory))

    def pull_project(self, git_project):
        print(f'Scanning: {self.repository_directory}/{git_project}\n'
              f'Pulling latest changes for {git_project}\n'
              f'{self.git_action(command="git pull", directory=os.path.normpath(os.path.join(self.repository_directory, git_project)))}')
        if self.set_to_default_branch:
            default_branch = self.git_action("git symbolic-ref refs/remotes/origin/HEAD",
                                             directory=f"{self.repository_directory}/{git_project}")
            default_branch = re.sub("refs/remotes/origin/", "", default_branch).strip()
            print(f"Checking out default branch ",
                  self.git_action(f'git checkout "{default_branch}"',
                                  directory=f"{self.repository_directory}/{git_project}"))


def usage():
    print(f"Usage: \n"
          f"-h | --help           [ See usage for script ]\n"
          f"-b | --default-branch [ Checkout default branch ]\n"
          f"-c | --clone          [ Clone projects specified  ]\n"
          f"-d | --directory      [ Directory to clone/pull projects ]\n"
          f"-f | --file           [ File with repository links   ]\n"
          f"-p | --pull           [ Pull projects in parent directory ]\n"
          f"-r | --repositories   [ Comma separated Git URLs ]\n"
          f"-t | --threads        [ Number of parallel threads - Default 4 ]\n"
          f"\n"
          f"repository-manager \n\t"
          f"--clone \n\t"
          f"--pull \n\t"
          f"--directory '/home/user/Downloads'\n\t"
          f"--file '/home/user/Downloads/repositories.txt' \n\t"
          f"--repositories 'https://github.com/Knucklessg1/media-downloader,https://github.com/Knucklessg1/genius-bot'\n\t"
          f"--threads 8")


def repository_manager(argv):
    gitlab = Git()
    projects = []
    default_branch_flag = False
    clone_flag = False
    pull_flag = False
    directory = os.curdir
    file = None
    repositories = None
    threads = os.cpu_count()
    try:
        opts, args = getopt.getopt(argv, "hbcpd:f:r:t:",
                                   ["help", "default-branch", "clone", "pull", "directory=", "file=", "repositories=",
                                    "threads="])
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
        elif opt in ("-t", "--threads"):
            threads = arg

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

    gitlab.set_threads(threads=threads)

    projects = list(dict.fromkeys(projects))

    gitlab.set_git_projects(projects)

    gitlab.set_default_branch(set_to_default_branch=default_branch_flag)

    if clone_flag:
        gitlab.clone_projects_in_parallel()
    if pull_flag:
        gitlab.pull_projects_in_parallel()


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
