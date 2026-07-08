import argparse
import hashlib
import json
import platform
import re
import sys
from pathlib import Path
from urllib.request import Request, urlopen

LATEST_RELEASE_API = "https://api.github.com/repos/cgohlke/geospatial-wheels/releases/latest"
GDAL_DEPENDENCY_RE = re.compile(r'(?m)^    "GDAL @ [^"]+",$')
WHEEL_RE = re.compile(r"^gdal-(?P<version>[^-]+)-(?P<python>cp\d+)-(?P<abi>cp\d+)-(?P<platform>[^.]+)\.whl$", re.IGNORECASE)


def default_python_tag():
    return f"cp{sys.version_info.major}{sys.version_info.minor}"


def default_platform_tag():
    machine = platform.machine().lower()
    if sys.platform == "win32" and machine in {"amd64", "x86_64"}:
        return "win_amd64"
    raise RuntimeError("The cgohlke GDAL wheel updater currently supports 64-bit Windows only.")


def fetch_json(url):
    request = Request(url, headers={"Accept": "application/vnd.github+json", "User-Agent": "merge-geotiff-updater"})
    with urlopen(request) as response:
        return json.load(response)


def sha256_url(url):
    digest = hashlib.sha256()
    request = Request(url, headers={"User-Agent": "merge-geotiff-updater"})
    with urlopen(request) as response:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def find_gdal_asset(release, python_tag, platform_tag):
    matches = []
    for asset in release.get("assets", []):
        match = WHEEL_RE.match(asset["name"])
        if not match:
            continue
        if match.group("python") == python_tag and match.group("abi") == python_tag and match.group("platform") == platform_tag:
            matches.append((match.group("version"), asset))

    if not matches:
        release_name = release.get("tag_name", "latest")
        raise RuntimeError(f"No GDAL wheel found in {release_name} for {python_tag}-{python_tag}-{platform_tag}.")

    return sorted(matches, key=lambda item: item[0])[-1][1]


def update_pyproject(pyproject_path, dependency):
    content = pyproject_path.read_text(encoding="utf-8")
    updated, count = GDAL_DEPENDENCY_RE.subn(f'    "{dependency}",', content)
    if count != 1:
        raise RuntimeError(f"Expected exactly one GDAL dependency line in {pyproject_path}.")
    pyproject_path.write_text(updated, encoding="utf-8")


def build_parser():
    parser = argparse.ArgumentParser(description="Update pyproject.toml to the latest cgohlke GDAL wheel for this Python.")
    parser.add_argument("--pyproject", type=Path, default=Path(__file__).resolve().parents[2] / "pyproject.toml")
    parser.add_argument("--python-tag", default=default_python_tag(), help="Wheel Python/ABI tag, e.g. cp311.")
    parser.add_argument("--platform-tag", default=None, help="Wheel platform tag. Defaults to win_amd64 on 64-bit Windows.")
    parser.add_argument("--dry-run", action="store_true", help="Print the dependency that would be written without editing pyproject.toml.")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    platform_tag = args.platform_tag or default_platform_tag()

    release = fetch_json(LATEST_RELEASE_API)
    asset = find_gdal_asset(release, args.python_tag, platform_tag)
    url = asset["browser_download_url"]
    digest = sha256_url(url)
    dependency = f"GDAL @ {url}#sha256={digest}"

    if args.dry_run:
        print(dependency)
        return

    update_pyproject(args.pyproject, dependency)
    print(f"Updated GDAL to {asset['name']} from {release['tag_name']}.")


if __name__ == "__main__":
    main()
