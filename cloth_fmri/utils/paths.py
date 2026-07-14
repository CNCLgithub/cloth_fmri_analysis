import json
from pathlib import Path


def get_script_info(file_path=None):
    if file_path is None:
        file_path = __file__

    file_path = Path(file_path)
    script_name = file_path.stem
    parent_folder_name = file_path.resolve().parent.name

    return script_name, parent_folder_name


def list_subdirs(directory):
    directory = Path(directory)
    return sorted([
        path.resolve()
        for path in directory.iterdir()
        if path.is_dir()
    ])


def list_files(directory):
    directory = Path(directory)
    return sorted([
        path.resolve()
        for path in directory.iterdir()
        if path.is_file()
    ])


def load_json_file(file_path):
    with open(file_path, "r") as file:
        return json.load(file)
