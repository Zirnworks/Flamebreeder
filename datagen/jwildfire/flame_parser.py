"""Parse JWildfire .flame XML files into Python data structures.

Handles both single <flame> files and multi-flame <flames> packs.
"""

import re
import xml.etree.ElementTree as ET
from pathlib import Path


def parse_palette(palette_elem: ET.Element) -> list[tuple[int, int, int]]:
    """Parse a <palette> element into a list of 256 (R, G, B) tuples."""
    hex_text = palette_elem.text.strip().replace("\n", "").replace(" ", "")
    colors = []
    for i in range(0, len(hex_text), 6):
        chunk = hex_text[i:i + 6]
        if len(chunk) == 6:
            r = int(chunk[0:2], 16)
            g = int(chunk[2:4], 16)
            b = int(chunk[4:6], 16)
            colors.append((r, g, b))
    # Pad or truncate to exactly 256
    while len(colors) < 256:
        colors.append(colors[-1] if colors else (0, 0, 0))
    return colors[:256]


def palette_to_hex(colors: list[tuple[int, int, int]], line_width: int = 72) -> str:
    """Convert a list of 256 (R, G, B) tuples back to hex string with line breaks."""
    hex_str = "".join(f"{r:02X}{g:02X}{b:02X}" for r, g, b in colors)
    lines = []
    for i in range(0, len(hex_str), line_width):
        lines.append(hex_str[i:i + line_width])
    return "\n".join(lines)


def parse_xform(elem: ET.Element) -> dict:
    """Parse an <xform> or <finalxform> element into a dictionary."""
    data = dict(elem.attrib)

    # Parse known numeric fields
    for key in ["weight", "color", "symmetry", "opacity"]:
        if key in data:
            try:
                data[key] = float(data[key])
            except ValueError:
                pass

    # Parse coefs (affine matrix)
    if "coefs" in data:
        data["coefs"] = [float(x) for x in data["coefs"].split()]

    # Parse post-affine
    if "post" in data:
        data["post"] = [float(x) for x in data["post"].split()]

    # Parse chaos (transition probabilities)
    if "chaos" in data:
        data["chaos"] = [float(x) for x in data["chaos"].split()]

    return data


def parse_flame(flame_elem: ET.Element) -> dict:
    """Parse a single <flame> element into a dictionary.

    Returns dict with keys:
        - attrs: all flame-level attributes
        - xforms: list of xform dicts
        - finalxform: optional finalxform dict
        - palette: list of 256 (R,G,B) tuples
        - raw_xml: the original element (for round-tripping unknown attrs)
    """
    flame = {
        "attrs": dict(flame_elem.attrib),
        "xforms": [],
        "finalxform": None,
        "palette": None,
    }

    for child in flame_elem:
        if child.tag == "xform":
            flame["xforms"].append(parse_xform(child))
        elif child.tag == "finalxform":
            flame["finalxform"] = parse_xform(child)
        elif child.tag == "palette":
            flame["palette"] = parse_palette(child)

    return flame


def parse_flame_file(path: Path | str) -> list[dict]:
    """Parse a .flame file and return a list of flame dicts.

    Handles both single <flame> and multi-flame <flames> wrapper formats.
    Also handles files with multiple root <flame> elements (no wrapper).
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")

    # Some .flame files have multiple root elements without a wrapper.
    # Wrap in a synthetic root to make it valid XML.
    # Also handle XML declaration if present.
    text = text.strip()
    if text.startswith("<?xml"):
        text = text.split("?>", 1)[1].strip()

    # Ensure wrapped in <flames>...</flames>
    if not text.startswith("<flames"):
        text = f"<flames>{text}</flames>"
    elif not text.rstrip().endswith("</flames>"):
        # Has opening <flames> but missing closing tag
        text = text.rstrip() + "\n</flames>"

    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        # Try to recover by fixing common issues
        # Some files have unescaped & or other XML issues
        text = text.replace("&", "&amp;")
        try:
            root = ET.fromstring(text)
        except ET.ParseError:
            return []

    flames = []
    for elem in root.iter("flame"):
        try:
            flame = parse_flame(elem)
            if flame["palette"] is not None and len(flame["xforms"]) > 0:
                flames.append(flame)
        except Exception:
            continue

    return flames


def collect_all_flames(root_dir: Path | str) -> list[tuple[Path, dict]]:
    """Recursively find and parse all .flame files under a directory.

    Returns list of (source_path, flame_dict) tuples.
    """
    root_dir = Path(root_dir)
    results = []

    for flame_path in sorted(root_dir.rglob("*.flame")):
        flames = parse_flame_file(flame_path)
        for flame in flames:
            results.append((flame_path, flame))

    return results
