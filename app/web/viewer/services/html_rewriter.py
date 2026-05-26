from __future__ import annotations

from bs4 import BeautifulSoup


def rewrite_choice_html(html: str, local_paths_in_order: list[str]) -> str:
    """Choice HTML stores `data-media-content-hash="pending:..."` even after fetch.

    Resolve by positionally pairing imgs with the choice's media_ids → local_paths.
    Imgs beyond the supplied list get marked `data-missing="true"`.
    """
    if not html:
        return html

    soup = BeautifulSoup(html, "html.parser")
    idx = 0
    for img in soup.find_all("img"):
        if not img.has_attr("data-media-content-hash"):
            continue
        del img["data-media-content-hash"]
        if idx < len(local_paths_in_order):
            img["src"] = f"/media/{local_paths_in_order[idx]}"
        else:
            img["data-missing"] = "true"
            img["alt"] = img.get("alt", "missing media")
        idx += 1
    return str(soup)
