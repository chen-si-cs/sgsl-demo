"""Build a root-layout update ZIP for an existing HF Space; never uploads it."""
import argparse
import json
from pathlib import Path
import zipfile


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Choose a new ZIP path; existing files are not overwritten")
    files = [root / name for name in ("app.py", "README.md", "THIRD_PARTY_NOTICES.md",
                                     "packages.txt", "requirements.txt")]
    files += sorted((root / "src").rglob("*.py"))
    featured = root / "examples" / "featured"
    manifest_path = featured / "manifest.json"
    files.append(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for entry in manifest["videos"]:
        for key in ("file", "processed_file", "report_file"):
            path = (featured / entry[key]).resolve()
            if path.parent != featured.resolve():
                raise ValueError("Featured file must be directly inside the featured folder")
            files.append(path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(root).as_posix())
    print(f"Created {args.output}: {len(files)} files, {args.output.stat().st_size:,} bytes")
    print("Extract and upload the contents at the existing Space root; retain existing models and examples.")


if __name__ == "__main__":
    main()
