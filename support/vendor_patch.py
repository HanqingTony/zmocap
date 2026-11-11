"""Point the vendored inference code at zmocap's own helpers.

The vendored ExMR sources were written against the reference application and
import a couple of small utilities from its private package. zmocap must not
depend on that package: we ship our own implementation (``support/prefetch.py``)
and rewrite the vendored imports to use it.

``patch()`` is idempotent, so it can run on every start.
"""
from __future__ import annotations

from pathlib import Path

__all__ = ["HELPER_MODULE", "REWRITES", "patch"]

# Module name the vendored code will import; dropped at the vendor root, which is
# already on sys.path when the vendored package is imported.
HELPER_MODULE = "zmocap_batch_prefetch"

REWRITES = {
    "from mocap.batch_prefetch import": f"from {HELPER_MODULE} import",
}


def patch(vendor_root, support_dir=None, quiet: bool = False) -> list:
    """Install our helper into ``vendor_root`` and rewrite stale imports.

    Returns the list of files that were modified.
    """
    vendor_root = Path(vendor_root)
    support_dir = Path(support_dir) if support_dir is not None else Path(__file__).resolve().parent
    helper = support_dir / "prefetch.py"
    if not helper.is_file():
        raise FileNotFoundError(f"缺少助手模块: {helper}")

    destination = vendor_root / f"{HELPER_MODULE}.py"
    content = helper.read_text(encoding="utf-8")
    if not destination.is_file() or destination.read_text(encoding="utf-8") != content:
        destination.write_text(content, encoding="utf-8")

    changed, rewritten = [], 0
    for path in vendor_root.rglob("*.py"):
        if path == destination:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        updated = text
        for old, new in REWRITES.items():
            if old in updated:
                rewritten += updated.count(old)
                updated = updated.replace(old, new)
        if updated != text:
            path.write_text(updated, encoding="utf-8")
            changed.append(path)

    if not quiet:
        print(f"[zmocap] vendor helper: {destination.name}; rewritten imports: {rewritten}")
    return changed
