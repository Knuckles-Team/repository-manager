# Repository Manager
*Version: 0.0.6*

Manage your Git projects

Run all Git supported tasks using Git Actions command

### Usage:
| Short Flag | Long Flag      | Description                       |
|------------|----------------|-----------------------------------|
| -h         | --help         | See Usage                         |
| -f         | --file         | File with repository links        |
| -c         | --clone        | Clone projects specified          |
| -p         | --pull         | Pull projects in parent directory |
| -d         | --directory    | Directory to clone/pull projects  |
| -r         | --repositories | Comma separated Git URLs          |

### Example:
```bash
git-manager --clone --pull --directory '/home/user/Downloads' --file '/home/user/Downloads/repositories.txt' --repositories 'https://github.com/Knucklessg1/media-downloader,https://github.com/Knucklessg1/genius-bot'
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
