import re

FOLDER_RE = re.compile(
    r"""
    ^(?P<scene>rotate|wind|drape|ball)
    _mass_(?P<mass>[0-9eE.+-]+)
    _bs_(?P<bs>[0-9eE.+-]+)
    _.*$
    """,
    re.VERBOSE,
)


def parse_folder_name(name: str):
    m = FOLDER_RE.match(name)
    if m is None:
        raise ValueError(f"Cannot parse folder name: {name}")

    d = m.groupdict()
    return {
        "scene": d["scene"],
        "mass": float(d["mass"]),
        "bs": float(d["bs"]),
    }