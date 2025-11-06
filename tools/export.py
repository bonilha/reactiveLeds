#!/usr/bin/env python3
# tools/export_for_chat.py
import sys, os, glob, hashlib, json, datetime, pathlib

# Quais arquivos entram no bundle (ajuste se precisar)
PATTERNS = [
    "sync.py",
    "fxcore.py",
    "effects/**/*.py",
    "config/**/*.json",
    "requirements.txt",
    "README.md",
]

def gather_files(patterns):
    seen, files = set(), []
    for pat in patterns:
        for p in glob.glob(pat, recursive=True):
            if os.path.isfile(p):
                rp = os.path.relpath(p).replace("\\", "/")
                if rp not in seen:
                    seen.add(rp)
                    files.append(rp)
    return sorted(files)

def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def emit_block(path):
    print(f"```file:path={path}")
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        print(f.read(), end="")
    print("\n```")
    print()

def main():
    root = pathlib.Path(".").resolve()
    files = gather_files(PATTERNS)
    if not files:
        print("# Nenhum arquivo correspondente aos padrões.", file=sys.stderr)
        sys.exit(1)

    meta = {
        "bundle_version": 1,
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "cwd": str(root),
        "files": [{"path": p, "sha256": sha256_of(p)} for p in files],
    }

    print("<!-- bundle:start -->")
    print("```bundle:meta")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print("```")
    print()
    for p in files:
        emit_block(p)
    print("<!-- bundle:end -->")

if __name__ == "__main__":
    main()