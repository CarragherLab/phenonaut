# Copyright © The University of Edinburgh, 2022.
# Development has been supported by GSK.

import json
from pathlib import Path
from typing import Union

import yaml


def check_path(
    p: Union[Path, str],
    is_dir: bool = False,
    make_parents: bool = True,
    make_dir_if_dir: bool = True,
) -> Path:
    """Check a user supplied path (str or Path), ensuring parents exist etc

    Parameters
    ----------
    p : Union[Path, str]
        File or directory path supplied by user
    is_dir : bool, optional
        If the path supplied by the user should be a directory, then set it as such by assigning is_dir to true, by default False
    make_parents : bool, optional
        If the parent directories of the supplied path do not exist, the make them, by default True
    make_dir_if_dir : bool, optional
        If the supplied path is a directory, but it does not exist, the make it, by default True

    Returns
    -------
    Path
        Path object pointing to the user supplied path, with parents made (if requested), and the directory itself made (if a directory and requested)

    Raises
    ------
    ValueError
        Parent does not exist, and make_parents was False
    ValueError
        Passed path was not a string or pathlib.Path
    """
    if isinstance(p, str):
        p = Path(p)
    if isinstance(p, Path):
        p = p.resolve()

        # Is dir
        if is_dir:
            if p.exists():
                return p
            if make_parents and not p.parent.exists():
                p.parent.mkdir(parents=True)
            if make_dir_if_dir and not p.exists():
                p.mkdir(parents=True)
            return p
        # Is file
        else:
            if not p.parent.exists():
                if make_parents:
                    p.mkdir(parents=True)
                    return p
                else:
                    raise ValueError(
                        f"{p.parent} does not exist, and make_parents was False when specifying the file {p}"
                    )
            else:
                return p
    else:
        raise ValueError(
            f"File/directory ({p}) passed as argument was not a string or Path, it was {type(p)}"
        )


def load_dict(file_path: Union[str, Path, None], cast_none_to_dict=False):
    if cast_none_to_dict and file_path is None:
        return {}
    if isinstance(file_path, dict):
        return file_path
    if isinstance(file_path, str):
        file_path = Path(file_path)
    if not isinstance(file_path, (str, Path)):
        raise TypeError(
            f"Argument 'file_path' to load_dict function was not None, str, or Path. It was {type(file_path)}"
        )
    if not file_path.exists():
        raise FileNotFoundError(f"The file '{file_path}' does not exist")
    if file_path.suffix in [".yml", ".yaml"]:
        return yaml.safe_load(open(file_path))
    elif file_path.suffix == ".json":
        return json.load(open(file_path))
    else:
        raise ValueError(
            f"File had invalid extension ('{file_path.suffix}'), please supply a .yml, .yaml, or .json file"
        )
