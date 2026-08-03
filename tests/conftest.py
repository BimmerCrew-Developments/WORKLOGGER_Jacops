import csv
import zipfile
from pathlib import Path

import pytest
from PIL import Image
from reportlab.pdfgen.canvas import Canvas


@pytest.fixture
def audit_zip(tmp_path):
    """Build a representative SafetyAuditor export without repository fixtures."""
    source = tmp_path / "fixture-source"
    source.mkdir()

    rows = [
        ("1", "", "datetime", "audit_completed", "1754136000000", "", "", ""),
        ("2", "", "text", "Project/Locatie Naam", "Fixture Project", "", "", ""),
        ("3", "", "text", "Building ID", "BLDG-42", "", "", ""),
        ("4", "", "address", "Adres", "", "Teststraat 42", "", ""),
        ("5", "", "text", "Postcode + Stad", "1234 AB Teststad", "", "", ""),
        ("6", "", "text", "Contactpersoon", "Ada Tester", "", "", ""),
        ("7", "", "text", "Quadrant", "Noord", "", "", ""),
        ("8", "", "text", "Gekoppelde kleur duct", "Blauw", "", "", ""),
        ("9", "", "text", "Hoeveel units gelast?", "7", "", "", ""),
        ("10", "", "section", "Gebruikte materialen", "", "", "", ""),
        ("11", "10", "number", "Kabel", "2", "", "", ""),
        ("12", "", "section", "Post Afmeldingen", "", "", "", ""),
        ("13", "12", "number", "Las geregistreerd", "3", "", "", ""),
        ("14", "", "media", "Inspectiefoto", "", "", "", "photo-a;photo-b"),
    ]
    csv_path = source / "representative.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["export generated for tests"])
        writer.writerow(["ID", "Parent ID", "Type", "Label", "Primary", "Secondary", "Note", "Media"])
        writer.writerows(rows)

    for name, color in (("photo-a.jpeg", "red"), ("photo-b.jpeg", "blue")):
        Image.new("RGB", (96, 72), color).save(source / name, quality=90)

    attachment = source / "inspection-attachment.pdf"
    canvas = Canvas(str(attachment))
    canvas.drawString(72, 760, "Generated fixture attachment")
    canvas.save()

    archive = tmp_path / "representative-audit.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in source.iterdir():
            bundle.write(path, f"export/{path.name}")
    return archive
