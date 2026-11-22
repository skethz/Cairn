#!/usr/bin/env python3
"""Verify package bytes and historical source identity without importing the model."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def source_identity():
    manifest = json.loads((ROOT / 'rust/evidence/optimization/release_manifest.json').read_text())
    rows = []
    for name, expected in sorted(manifest.items()):
        if name.startswith(('src/', 'native/', 'configs/')) or name in {'Cargo.toml', 'Cargo.lock', 'build.rs', 'build_offline.sh'}:
            path = ROOT / 'rust' / name
            actual = digest(path) if path.is_file() else None
            rows.append(dict(path='rust/' + name, expected_sha256=expected,
                             actual_sha256=actual, matches=actual == expected))
    extras = sorted(str(p.relative_to(ROOT)) for folder in ('src', 'native', 'configs')
                    for p in (ROOT / 'rust' / folder).rglob('*')
                    if p.is_file() and str(p.relative_to(ROOT / 'rust')) not in manifest)
    return dict(reference='rust/evidence/optimization/release_manifest.json',
                matched=sum(row['matches'] for row in rows), total=len(rows),
                extra_files=extras, files=rows)

def main():
    failures = []
    for line in (ROOT / 'MANIFEST.tsv').read_text().splitlines()[1:]:
        name, size, sha = line.split('\t')
        path = ROOT / name
        if not path.is_file() or path.stat().st_size != int(size) or digest(path) != sha:
            failures.append(name)
    identity = source_identity()
    if identity != json.loads((ROOT / 'docs/source_identity.json').read_text()):
        failures.append('source identity report no longer matches current files')
    if failures:
        raise SystemExit('Verification failed: ' + ', '.join(failures))
    print('Package hashes: PASS')
    print(f"Historical source identity: {identity['matched']}/{identity['total']} match")
    for row in identity['files']:
        if not row['matches']:
            print('Documented difference: ' + row['path'])
    print('No training, model loading, or native build performed.')

if __name__ == '__main__':
    main()
