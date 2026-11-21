"""Extract page-marked paper text, links, and rendered page evidence."""

import argparse
import hashlib
import json
from pathlib import Path

import fitz


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    source = options.pdf.resolve()
    output = options.output.resolve()
    pages_dir = output / "paper_pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    document = fitz.open(source)
    text_parts, pages, links = [], [], []
    for index, page in enumerate(document):
        number = index + 1
        text = page.get_text(sort=True)
        text_parts.extend([f"\n{'=' * 24} PDF PAGE {number:02d} {'=' * 24}\n", text])
        page.get_pixmap(matrix=fitz.Matrix(1.6, 1.6), alpha=False).save(
            pages_dir / f"page_{number:02d}.png")
        page_links = [{key: str(value) if key in {"from", "to"} else value
                       for key, value in item.items()} for item in page.get_links()]
        links.extend([{"pdf_page": number, **item} for item in page_links])
        pages.append({"pdf_page": number, "text_characters": len(text),
                      "render": f"paper_pages/page_{number:02d}.png"})
    (output / "paper_text.txt").write_text("\n".join(text_parts), encoding="utf-8")
    inventory = {"source_pdf": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                 "page_count": len(document), "metadata": document.metadata,
                 "pages": pages, "links": links}
    (output / "paper_source.json").write_text(json.dumps(inventory, indent=2, ensure_ascii=False),
                                               encoding="utf-8")
    print(json.dumps({"pages": len(document), "links": len(links), "output": str(output)},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
