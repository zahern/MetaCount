from __future__ import annotations

import argparse
from pathlib import Path


MARKER_TO_ASSET = {
    "FIT_OVERVIEW": "quickstart_fit_overview.md",
    "FIT_COEFFICIENTS": "quickstart_fit_coefficients.md",
    "BATCH_RESULTS": "batch_results_table.md",
    "BOOK_MANUAL_DIFF": "scoping_cmf_vs_traditional_differences.md",
}


def _load_asset_text(asset_path: Path) -> str:
    if asset_path.exists():
        return asset_path.read_text(encoding="utf-8").strip()
    return f"Missing asset: {asset_path.as_posix()}"


def _replace_marker_block(deck_text: str, marker: str, replacement: str) -> tuple[str, bool]:
    start_marker = f"<!-- AUTO_TABLE:{marker}:start -->"
    end_marker = f"<!-- AUTO_TABLE:{marker}:end -->"

    start_idx = deck_text.find(start_marker)
    end_idx = deck_text.find(end_marker)

    if start_idx == -1 or end_idx == -1 or end_idx < start_idx:
        return deck_text, False

    before = deck_text[: start_idx + len(start_marker)]
    after = deck_text[end_idx:]
    updated = before + "\n\n" + replacement + "\n\n" + after
    return updated, True


def update_deck(deck_path: Path, assets_dir: Path) -> tuple[int, int]:
    text = deck_path.read_text(encoding="utf-8")
    replaced = 0

    for marker, filename in MARKER_TO_ASSET.items():
        asset_text = _load_asset_text(assets_dir / filename)
        text, ok = _replace_marker_block(text, marker, asset_text)
        if ok:
            replaced += 1

    deck_path.write_text(text, encoding="utf-8")
    return replaced, len(MARKER_TO_ASSET)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inject generated markdown tables into cmf_vs_count_comparison_slides.md markers."
    )
    parser.add_argument(
        "--deck",
        default="cmf_vs_count_comparison_slides.md",
        help="Path to the Marp slide markdown file.",
    )
    parser.add_argument(
        "--assets-dir",
        default="results/slide_assets",
        help="Directory that contains generated .md assets from tutorial notebooks.",
    )

    args = parser.parse_args()

    deck_path = Path(args.deck)
    assets_dir = Path(args.assets_dir)

    if not deck_path.exists():
        raise FileNotFoundError(f"Deck not found: {deck_path}")

    replaced, total = update_deck(deck_path=deck_path, assets_dir=assets_dir)

    print(f"Updated {deck_path.resolve()}")
    print(f"Replaced marker blocks: {replaced}/{total}")
    print(f"Assets directory: {assets_dir.resolve()}")


if __name__ == "__main__":
    main()
