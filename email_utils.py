#274
import smtplib
import os
import re as _re
import logging
import base64
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


def _md_to_html(md: str) -> str:
    """
    Convert markdown to HTML, stripping all image tags.
    Images are handled separately as email attachments,
    so we don't depend on the LLM writing correct image paths.
    """
    lines = md.split("\n")
    html_lines = []
    in_ul = False

    for line in lines:
        # Strip all image markdown completely — we handle images as attachments
        line = _re.sub(r"!\[.*?\]\([^)]+\)", "", line).strip()

        if line.startswith("### "):
            if in_ul: html_lines.append("</ul>"); in_ul = False
            html_lines.append(f"<h3>{line[4:].strip()}</h3>")
        elif line.startswith("## "):
            if in_ul: html_lines.append("</ul>"); in_ul = False
            html_lines.append(f"<h2>{line[3:].strip()}</h2>")
        elif line.startswith("# "):
            if in_ul: html_lines.append("</ul>"); in_ul = False
            html_lines.append(f"<h1>{line[2:].strip()}</h1>")
        elif line.startswith("- ") or line.startswith("* "):
            if not in_ul: html_lines.append("<ul>"); in_ul = True
            item = _re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", line[2:])
            html_lines.append(f"<li>{item}</li>")
        elif line in ("", "---"):
            if in_ul: html_lines.append("</ul>"); in_ul = False
            html_lines.append("<br>")
        else:
            if in_ul: html_lines.append("</ul>"); in_ul = False
            formatted = _re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", line)
            if formatted.strip():
                html_lines.append(f"<p>{formatted}</p>")

    if in_ul:
        html_lines.append("</ul>")
    return "\n".join(html_lines)


def _collect_image_paths(image_dir: str | None) -> list[Path]:
    """
    Return a sorted list of image Paths found in the session image folder.
    Returns an empty list if the folder doesn't exist or has no images.
    """
    if not image_dir:
        return []

    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
    img_folder = Path("images") / image_dir

    if not img_folder.exists():
        logger.warning(f"Image folder not found: {img_folder}")
        return []

    paths = sorted(
        p for p in img_folder.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    return paths


def _build_inline_charts_html(image_paths: list[Path]) -> str:
    """
    Build an HTML section that references each chart via a cid: Content-ID.

    Instead of embedding a giant base64 blob inside the HTML (the old way),
    we write  <img src="cid:chart_0">  and attach the actual image file as a
    MIME part with that same Content-ID.  The email client stitches them
    together — the result looks identical to the reader but the email is
    structured correctly and passes more spam filters.

    Returns the HTML string (empty string if no images).
    """
    if not image_paths:
        return ""

    img_tags = "".join(
        f'<div style="margin:12px 0;">'
        f'<img src="cid:chart_{i}" alt="Chart {i + 1}" '
        f'style="max-width:100%;border-radius:6px;border:1px solid #e2e8f0;">'
        f'</div>'
        for i, _ in enumerate(image_paths)
    )

    return (
        "<h2 style='color:#1e40af;border-bottom:2px solid #e2e8f0;padding-bottom:6px;'>"
        "📊 Charts</h2>\n"
        + img_tags
    )


def send_email_report(
    to_email: str,
    subject: str,
    report_markdown: str,
    image_dir: str | None = None,
):
    """
    Send a markdown report via email with charts as proper attachments.

    HOW IMAGES WORK (the change from the old version):
    ────────────────────────────────────────────────────
    Old way  →  images were base64-encoded and pasted directly inside the
                HTML as giant data: URIs.  Problem: some email clients
                (Outlook, Gmail) block or strip inline base64 images for
                security reasons, so charts often didn't show up.

    New way  →  images are attached as real MIME parts (like paper clips),
                each given a unique Content-ID (e.g. "chart_0").  The HTML
                body references them with  <img src="cid:chart_0">.  The
                email client links them automatically — it's the standard
                way professional emails embed images, and it works in every
                major email client including Outlook and Gmail.

    The email structure is now:
        multipart/related          ← ties HTML body + inline images together
        ├── multipart/alternative  ← plain-text fallback + HTML version
        │   ├── text/plain         ← readable if client strips HTML
        │   └── text/html          ← main rendered view (references cid:)
        └── image/png  (cid:chart_0)   ← actual chart files
        └── image/png  (cid:chart_1)
        └── ...

    Args:
        to_email:         Recipient address.
        subject:          Email subject.
        report_markdown:  The full markdown report text.
        image_dir:        Session image subfolder name inside images/.
                          E.g. "20240524_120000_abc123".

    Configure via .env:
        SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, SMTP_FROM
    """
    smtp_host     = os.getenv("SMTP_HOST")
    smtp_port     = int(os.getenv("SMTP_PORT", 587))
    smtp_user     = os.getenv("SMTP_USER")
    smtp_password = os.getenv("SMTP_PASSWORD")
    smtp_from     = os.getenv("SMTP_FROM", smtp_user)

    if not all([smtp_host, smtp_user, smtp_password]):
        raise ValueError("SMTP configuration is missing in .env")

    # ── 1. Collect image files from disk ────────────────────────────────────
    image_paths = _collect_image_paths(image_dir)
    n_charts = len(image_paths)

    # ── 2. Convert markdown → HTML (LLM chart section stripped out) ─────────
    clean_md = _re.sub(
        r"## Charts.*?(?=## |\Z)",
        "",
        report_markdown,
        flags=_re.DOTALL | _re.IGNORECASE,
    ).strip()

    report_html_body = _md_to_html(clean_md)

    # ── 3. Build chart section using cid: references ─────────────────────────
    charts_html = _build_inline_charts_html(image_paths)

    # ── 4. Assemble the full HTML document ───────────────────────────────────
    html_content = f"""
<html>
<head>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            line-height: 1.7; color: #1a1a2e;
            max-width: 740px; margin: 0 auto; padding: 20px;
        }}
        h1, h2, h3 {{
            color: #1e40af;
            border-bottom: 2px solid #e2e8f0;
            padding-bottom: 6px;
        }}
        ul {{ padding-left: 1.4em; }}
        li {{ margin-bottom: 4px; }}
        p  {{ margin: 6px 0; }}
        .wrapper {{
            border: 1px solid #e2e8f0; border-radius: 10px;
            padding: 24px 28px; margin-top: 16px; background: #fafcff;
        }}
        .footer {{
            margin-top: 28px; font-size: 0.78rem; color: #888;
            border-top: 1px solid #e2e8f0; padding-top: 10px;
        }}
    </style>
</head>
<body>
    <p>Hello,</p>
    <p>Here is your analysis report from <strong>AutoAnalyst</strong>:</p>
    <div class="wrapper">
        {report_html_body}
        {charts_html}
    </div>
    <div class="footer">
        Sent by AutoAnalyst — AI-powered CSV Analysis Agent.
        {f'({n_charts} chart(s) attached)' if n_charts else ''}
    </div>
</body>
</html>
"""

    # ── 5. Build the MIME structure ───────────────────────────────────────────
    #
    # multipart/related  →  groups the HTML with its inline image attachments
    #   multipart/alternative  →  plain text fallback + HTML version
    #     text/plain
    #     text/html
    #   image/png  (Content-ID: chart_0)
    #   image/png  (Content-ID: chart_1)
    #   ...
    #
    msg_related = MIMEMultipart("related")
    msg_related["Subject"] = subject
    msg_related["From"]    = smtp_from
    msg_related["To"]      = to_email

    # Inner alternative part (plain text + HTML)
    msg_alternative = MIMEMultipart("alternative")
    msg_alternative.attach(MIMEText(report_markdown, "plain"))
    msg_alternative.attach(MIMEText(html_content,    "html"))
    msg_related.attach(msg_alternative)

    # Attach each chart image as a MIME part with a matching Content-ID
    for i, img_path in enumerate(image_paths):
        try:
            raw = img_path.read_bytes()
            suffix = img_path.suffix.lower().lstrip(".")
            mime_subtype = "jpeg" if suffix == "jpg" else suffix  # e.g. "png"

            img_part = MIMEImage(raw, _subtype=mime_subtype)
            # Content-ID must match the cid: used in the HTML above
            img_part.add_header("Content-ID", f"<chart_{i}>")
            # Content-Disposition: inline tells the client to render it in the body
            img_part.add_header(
                "Content-Disposition", "inline",
                filename=f"chart_{i + 1}{img_path.suffix}"
            )
            msg_related.attach(img_part)
            logger.info(f"Email: attached {img_path.name} as chart_{i} ({len(raw)//1024} KB)")
        except Exception as e:
            logger.warning(f"Could not attach image {img_path.name}: {e}")

    # ── 6. Send ───────────────────────────────────────────────────────────────
    try:
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.send_message(msg_related)
        logger.info(f"Email sent to {to_email} ({n_charts} charts attached)")
        return True
    except Exception as e:
        logger.error(f"Failed to send email: {e}")
        raise
