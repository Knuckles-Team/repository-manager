# Repository Manager

![PyPI - Version](https://img.shields.io/pypi/v/repository-manager)
![PyPI - Downloads](https://img.shields.io/pypi/dd/repository-manager)
![GitHub Repo stars](https://img.shields.io/github/stars/Knuckles-Team/repository-manager)
![GitHub forks](https://img.shields.io/github/forks/Knuckles-Team/repository-manager)
![GitHub contributors](https://img.shields.io/github/contributors/Knuckles-Team/repository-manager)
![PyPI - License](https://img.shields.io/pypi/l/repository-manager)
![GitHub](https://img.shields.io/github/license/Knuckles-Team/repository-manager)

![GitHub last commit (by committer)](https://img.shields.io/github/last-commit/Knuckles-Team/repository-manager)
![GitHub pull requests](https://img.shields.io/github/issues-pr/Knuckles-Team/repository-manager)
![GitHub closed pull requests](https://img.shields.io/github/issues-pr-closed/Knuckles-Team/repository-manager)
![GitHub issues](https://img.shields.io/github/issues/Knuckles-Team/repository-manager)

![GitHub top language](https://img.shields.io/github/languages/top/Knuckles-Team/repository-manager)
![GitHub language count](https://img.shields.io/github/languages/count/Knuckles-Team/repository-manager)
![GitHub repo size](https://img.shields.io/github/repo-size/Knuckles-Team/repository-manager)
![GitHub repo file count (file type)](https://img.shields.io/github/directory-file-count/Knuckles-Team/repository-manager)
![PyPI - Wheel](https://img.shields.io/pypi/wheel/repository-manager)
![PyPI - Implementation](https://img.shields.io/pypi/implementation/repository-manager)

*Version: 0.5.0*

Manage your Git projects

Run all Git supported tasks using Git Actions command

This repository is actively maintained - Contributions are welcome!

<details>
  <summary><b>Usage:</b></summary>

| Short Flag | Long Flag        | Description                            |
|------------|------------------|----------------------------------------|
| -h         | --help           | See Usage                              |
| -b         | --default-branch | Checkout default branch                |
| -c         | --clone          | Clone projects specified               |
| -d         | --directory      | Directory to clone/pull projects       |
| -f         | --file           | File with repository links             |
| -p         | --pull           | Pull projects in parent directory      |
| -r         | --repositories   | Comma separated Git URLs               |
| -t         | --threads        | Number of parallel threads - Default 4 |

</details>

<details>
  <summary><b>Example:</b></summary>

```bash
repository-manager \
    --clone  \
    --pull  \
    --directory '/home/user/Downloads'  \
    --file '/home/user/Downloads/repositories.txt'  \
    --repositories 'https://github.com/Knucklessg1/media-downloader,https://github.com/Knucklessg1/genius-bot' \
    --threads 8
```

</details>

<details>
  <summary><b>Installation Instructions:</b></summary>

Install Python Package

```bash
python -m pip install repository-manager
```
</details>

<details>
  <summary><b>Repository Owners:</b></summary>


<img width="100%" height="180em" src="https://github-readme-stats.vercel.app/api?username=Knucklessg1&show_icons=true&hide_border=true&&count_private=true&include_all_commits=true" />

![GitHub followers](https://img.shields.io/github/followers/Knucklessg1)
![GitHub User's stars](https://img.shields.io/github/stars/Knucklessg1)
</details>
