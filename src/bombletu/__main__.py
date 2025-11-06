"""使包可通过 `python -m bombletu` 直接运行。"""

from .app import run


def main() -> None:
    run()


if __name__ == "__main__":
    main()
