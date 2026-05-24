from __future__ import annotations

"""Public repository entrypoint for the local ControlLight web demo."""

from ControlLight.bootstrap import bootstrap_local_paths


def main() -> None:
    bootstrap_local_paths()
    from ControlLight.web_demo import main as demo_main

    demo_main()


if __name__ == "__main__":
    main()
