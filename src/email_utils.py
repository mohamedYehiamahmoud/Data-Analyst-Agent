import smtplib
import os
import logging
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

def send_email_report(to_email: str, subject: str, report_markdown: str):
    """
    Send a markdown report via email using SMTP.
    Configured via environment variables:
    - SMTP_HOST
    - SMTP_PORT
    - SMTP_USER
    - SMTP_PASSWORD
    - SMTP_FROM
    """
    smtp_host = os.getenv("SMTP_HOST")
    smtp_port = int(os.getenv("SMTP_PORT", 587))
    smtp_user = os.getenv("SMTP_USER")
    smtp_password = os.getenv("SMTP_PASSWORD")
    smtp_from = os.getenv("SMTP_FROM", smtp_user)

    if not all([smtp_host, smtp_user, smtp_password]):
        logger.error("SMTP configuration is missing. Cannot send email.")
        raise ValueError("SMTP configuration is missing in .env")

    # Simple Markdown to HTML conversion (very basic)
    report_html_body = report_markdown.replace("\n", "<br>")

    html_content = f"""
    <html>
    <head>
        <style>
            body {{ font-family: sans-serif; line-height: 1.6; color: #333; }}
            h2 {{ color: #1e40af; border-bottom: 2px solid #e2e8f0; padding-bottom: 5px; }}
            pre {{ background: #f8fafc; padding: 10px; border-radius: 5px; border: 1px solid #e2e8f0; }}
            code {{ font-family: monospace; color: #e11d48; }}
        </style>
    </head>
    <body>
        <p>Hello,</p>
        <p>Here is your analysis report from <strong>AutoAnalyst</strong>:</p>
        <div style="margin-top: 20px; padding: 20px; border: 1px solid #e2e8f0; border-radius: 8px;">
            {report_html_body}
        </div>
        <p style="margin-top: 30px; font-size: 0.8rem; color: #666;">
            Sent by AutoAnalyst Agent.
        </p>
    </body>
    </html>
    """

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = smtp_from
    msg["To"] = to_email

    msg.attach(MIMEText(report_markdown, "plain"))
    msg.attach(MIMEText(html_content, "html"))

    try:
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.send_message(msg)
        logger.info(f"Email sent successfully to {to_email}")
        return True
    except Exception as e:
        logger.error(f"Failed to send email: {e}")
        raise e
