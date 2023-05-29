import argparse
import subprocess
import re
import os
import sys

from typing import Tuple
from pathlib import Path
from setuptools_scm import get_version


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


if __name__ == "__main__":
    args = define_parser().parse_args()

    version_parts = get_version(
        root=".", relative_to=__file__, version_scheme="no-guess-dev"
    ).split(".")
    major = int(version_parts[0])
    minor = int(version_parts[1])
    patch = int(version_parts[2])

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

    version_str = "v{:d}.{:d}.{:d}".format(major, minor, patch)
    if args.skip_increment:
        print("Skipping incrementing of version number!")
        print("Removing old version tag")
        cmd_str = f"git tag -d {version_str}"
        print(cmd_str)
        subprocess.run(cmd_str, shell=True)

    print("Releasing version: {:s}".format(version_str[1:]))

    cmd_str = "git tag -a {:s} -m '{:s}'".format(version_str, args.message)
    print(cmd_str)
    subprocess.run(cmd_str, shell=True)

    cmd_str = "git push origin {:s}".format(version_str)
    subprocess.run(cmd_str, shell=True)
