from __future__ import annotations

"""Public repository entrypoint for ControlLight prediction."""

from ControlLight.bootstrap import bootstrap_local_paths


def main() -> None:
    bootstrap_local_paths()
    from ControlLight.predict import main as predict_main

    predict_main()


if __name__ == "__main__":
    main()
