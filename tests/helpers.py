from pathlib import Path


def write_pdf(path: Path, page_texts: list[str]) -> None:
    objects: dict[int, str] = {
        1: "<< /Type /Catalog /Pages 2 0 R >>",
        4: "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    }
    page_ids: list[int] = []
    next_object_id = 5

    for text in page_texts:
        page_id = next_object_id
        contents_id = next_object_id + 1
        page_ids.append(page_id)
        objects[page_id] = (
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            "/Resources << /Font << /F1 4 0 R >> >> "
            f"/Contents {contents_id} 0 R >>"
        )
        stream = f"BT /F1 12 Tf 72 720 Td ({_escape_pdf_text(text)}) Tj ET"
        objects[contents_id] = (
            f"<< /Length {len(stream.encode('utf-8'))} >>\nstream\n{stream}\nendstream"
        )
        next_object_id += 2

    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
    objects[2] = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>"
    objects[3] = "<< /Producer (huginn-tests) >>"

    buffer = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    max_object_id = max(objects)
    for object_id in range(1, max_object_id + 1):
        body = objects[object_id]
        offsets.append(len(buffer))
        buffer.extend(f"{object_id} 0 obj\n{body}\nendobj\n".encode("utf-8"))

    xref_offset = len(buffer)
    buffer.extend(f"xref\n0 {max_object_id + 1}\n".encode("utf-8"))
    buffer.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        buffer.extend(f"{offset:010} 00000 n \n".encode("utf-8"))
    buffer.extend(
        (
            f"trailer\n<< /Size {max_object_id + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("utf-8")
    )
    path.write_bytes(buffer)


def _escape_pdf_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
