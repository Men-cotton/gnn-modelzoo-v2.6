#!/usr/bin/env python3
"""Print Python build paths used by setup.sh."""

import sysconfig


def main() -> int:
    include = sysconfig.get_path("include") or ""
    prefix = sysconfig.get_config_var("prefix") or ""
    print(f"include={include}")
    print(f"prefix={prefix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
