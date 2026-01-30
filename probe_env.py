import json

from probe_lib import probe_env


def main():
    print(json.dumps(probe_env(), indent=2))


if __name__ == "__main__":
    main()
