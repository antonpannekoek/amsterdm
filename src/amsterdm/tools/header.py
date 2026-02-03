import argparse
from pathlib import Path

from amsterdm.candidate import openfile


def run(filename: str | Path):
    with openfile(filename) as candidate:
        print(f"{filename}:")
        for key, value in candidate.header.items():
            print(f" {key:<8s} = {value}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="Input file name")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    run(args.filename)


if __name__ == "__main__":
    main()
