# Repository Manager
*Version: 0.4.0*

Manage your Git projects

Run all Git supported tasks using Git Actions command

### Usage:
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

### Example:
```bash
repository-manager \
    --clone  \
    --pull  \
    --directory '/home/user/Downloads'  \
    --file '/home/user/Downloads/repositories.txt'  \
    --repositories 'https://github.com/Knucklessg1/media-downloader,https://github.com/Knucklessg1/genius-bot' \
    --threads 8
```

#### Install Instructions
Install Python Package

```bash
python -m pip install repository-manager
```

#### Build Instructions
Build Python Package

```bash
sudo chmod +x ./*.py
sudo pip install .
python3 setup.py bdist_wheel --universal
# Test Pypi
twine upload --repository-url https://test.pypi.org/legacy/ dist/* --verbose -u "Username" -p "Password"
# Prod Pypi
twine upload dist/* --verbose -u "Username" -p "Password"
```
