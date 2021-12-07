import os
from pathlib import Path


def files_in_dir(path):
    return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]


def create_dir_if_necessary(dir_path):
    Path(dir_path).mkdir(parents=True, exist_ok=True)


def create_file_dir_if_necessary(file_path):
    path, _ = os.path.split(file_path)
    create_dir_if_necessary(path)
