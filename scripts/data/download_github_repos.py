import subprocess
from tqdm import tqdm

from utils.file import load_json

projects = load_json('W:/Data/treevul_cppcfiltered_projects.json')

for project in tqdm(projects):
    cmd = f'git clone https://github.com/{project}.git'
    subprocess.run(cmd, shell=True, check=True)


