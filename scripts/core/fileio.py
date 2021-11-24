import os


def files_in_dir(path):
    return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
