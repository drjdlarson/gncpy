import argparse
import subprocess
import re
import os
import sys

from typing import Tuple
from pathlib import Path


VERSION_FILE = os.path.join(os.path.dirname(__file__), "pyproject.toml")


def get_active_branch_name():
    head_dir = Path(".") / ".git" / "HEAD"
    with head_dir.open("r") as f:
        content = f.read().splitlines()

    for line in content:
        if line[0:4] == "ref:":
            return line.partition("refs/heads/")[2]


def define_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Automatically create and push tag so CI/CD pipeline publishes a release"
    )

    choices = ("major", "minor", "patch")
    p.add_argument(
        "type",
        type=str,
        help="Type of release to perform",
        choices=choices,
    )

    p.add_argument(
        "-s",
        "--skip-increment",
        action="store_true",
        help="Flag indicating if incrementing the version should be skipped. "
        + "If this is passed then the type is irrelevant.",
    )

    default = "Automatic release"
    p.add_argument(
        "-m",
        "--message",
        type=str,
        default=default,
        help="Message for the release (Doesn't seem to be used in CI/CD). The default is: {:s}".format(
            default
        ),
    )

    return p


def get_match(line: str):
    return re.search('version\s*=\s*"(\d+).(\d+).(\d+)"', line)


def get_version() -> Tuple[int, int, int]:
    with open(VERSION_FILE, "r") as fin:
        for line in fin:
            matched = get_match(line)
            if matched:
                major = int(matched.groups()[0])
                minor = int(matched.groups()[1])
                patch = int(matched.groups()[2])
                return major, minor, patch

    raise RuntimeError("Failed to extract version from {:s}".format(VERSION_FILE))


def set_version(major: int, minor: int, patch: int):
    tmp_file = VERSION_FILE + ".tmp"
    with open(VERSION_FILE, "r") as fin:
        with open(tmp_file, "w") as fout:
            for line in fin:
                matched = get_match(line)
                if matched:
                    ind = line.find('"')
                    new_line = line[:ind] + '"{:d}.{:d}.{:d}"\n'.format(
                        major, minor, patch
                    )
                    fout.write(new_line)
                else:
                    fout.write(line)

    os.replace(tmp_file, VERSION_FILE)


if __name__ == "__main__":
    args = define_parser().parse_args()

    major, minor, patch = get_version()
    print("Current version: {:d}.{:d}.{:d}".format(major, minor, patch))

    cur_branch = get_active_branch_name()
    if cur_branch.lower() != "master":
        print(
            "WARN: Not on master branch ({:s}), checkout to master branch for release".format(
                cur_branch
            )
        )
        sys.exit(-1)

    if not args.skip_increment:
        if args.type == "major":
            major += 1
            minor = 0
            patch = 0
        elif args.type == "minor":
            minor += 1
            patch = 0
        elif args.type == "patch":
            patch += 1
        else:
            raise RuntimeError("Invalid type: {} should not be here".format(args.type))

        set_version(major, minor, patch)

    else:
        print("Skipping incrementing of version number!")

    version_str = "v{:d}.{:d}.{:d}".format(major, minor, patch)
    print("Releasing version: {:s}".format(version_str[1:]))

    if not args.skip_increment:
        cmd_str = "git add -u && git commit -m 'bump version' && git push"
        subprocess.run(cmd_str, shell=True)

    cmd_str = "git tag -a {:s} -m '{:s}'".format(version_str, args.message)
    subprocess.run(cmd_str, shell=True)

    cmd_str = "git push origin {:s}".format(version_str)
    subprocess.run(cmd_str, shell=True)