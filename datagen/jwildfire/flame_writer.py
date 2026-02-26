"""Write mutated flame dicts back to JWildfire .flame XML format."""

from pathlib import Path

from .flame_parser import palette_to_hex


def _format_value(key: str, value) -> str:
    """Format a value for XML attribute output."""
    if isinstance(value, list):
        return " ".join(str(v) for v in value)
    return str(value)


def xform_to_xml(xform: dict, tag: str = "xform", indent: str = "  ") -> str:
    """Convert an xform dict back to XML string."""
    parts = [f"{indent}<{tag}"]
    for key, value in xform.items():
        parts.append(f' {key}="{_format_value(key, value)}"')
    parts.append("/>")
    return "".join(parts)


def flame_to_xml(flame: dict) -> str:
    """Convert a flame dict back to a complete .flame XML string."""
    lines = []

    # Opening <flame> tag with all attributes
    attrs_str = " ".join(
        f'{k}="{v}"' for k, v in flame["attrs"].items()
    )
    lines.append(f'<flame {attrs_str}>')

    # xforms
    for xform in flame["xforms"]:
        lines.append(xform_to_xml(xform, tag="xform"))

    # finalxform
    if flame.get("finalxform"):
        lines.append(xform_to_xml(flame["finalxform"], tag="finalxform"))

    # palette
    if flame.get("palette"):
        hex_data = palette_to_hex(flame["palette"])
        lines.append(f'  <palette count="256" format="RGB">')
        lines.append(hex_data)
        lines.append("  </palette>")

    lines.append("</flame>")
    return "\n".join(lines)


def write_flame_file(flames: list[dict], path: Path | str) -> None:
    """Write one or more flame dicts to a .flame file.

    If multiple flames, wraps in <flames> root element.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    xml_parts = []
    for flame in flames:
        xml_parts.append(flame_to_xml(flame))

    if len(xml_parts) == 1:
        content = xml_parts[0]
    else:
        content = "<flames>\n" + "\n".join(xml_parts) + "\n</flames>"

    path.write_text(content, encoding="utf-8")
