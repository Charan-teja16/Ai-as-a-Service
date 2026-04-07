"""Email service for OTP and report sending."""
import os
import smtplib
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path


class EmailService:
    """Simple email service with graceful fallback."""

    def __init__(self):
        self.smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.sender_email = os.getenv("SMTP_SENDER", "")
        self.sender_password = os.getenv("SMTP_PASSWORD", "")
        self.enabled = os.getenv("SMTP_ENABLED", "false").lower() == "true" and bool(
            self.sender_email and self.sender_password
        )

    def _send(self, msg: MIMEMultipart) -> bool:
        if not self.enabled:
            print("[EMAIL SERVICE] Email sending disabled; message would be sent.")
            return True
        try:
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)
            return True
        except Exception as exc:
            print(f"Failed to send email: {exc}")
            return False

    def send_otp(self, to_email: str, otp_code: str) -> bool:
        """Send OTP code to email."""
        msg = MIMEMultipart()
        msg["From"] = self.sender_email or "no-reply@example.com"
        msg["To"] = to_email
        msg["Subject"] = "Password Reset OTP - AI-as-a-Service"
        body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; color: #111;">
            <p>Your password reset OTP code is:</p>
            <p style="font-size: 20px; font-weight: bold; color: #2563eb;">{otp_code}</p>
            <p>This code expires in 10 minutes. If you didn't request this, you can ignore this email.</p>
        </body>
        </html>
        """
        msg.attach(MIMEText(body, "html"))
        return self._send(msg)

    def send_report(self, to_email: str, report_path: Path, user_name: str, dataset_name: str | None = None, best_model: str | None = None, key_metric: str | None = None) -> bool:
        """Send PDF report to email with a concise, professional template."""
        msg = MIMEMultipart()
        msg["From"] = self.sender_email or "no-reply@example.com"
        msg["To"] = to_email
        msg["Subject"] = "Your ML Training Report - AI-as-a-Service"

        summary_lines = []
        if dataset_name:
            summary_lines.append(f"Dataset: <strong>{dataset_name}</strong>")
        if best_model:
            summary_lines.append(f"Best model: <strong>{best_model}</strong>")
        if key_metric:
            summary_lines.append(f"Key metric: <strong>{key_metric}</strong>")
        summary_html = "<br/>".join(summary_lines) if summary_lines else "Your report is attached."

        body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; color: #111;">
            <p>Hello {user_name or 'there'},</p>
            <p>Your machine learning training report is ready.</p>
            <p>{summary_html}</p>
            <p>You can download the full PDF from the attachment or directly in the platform.</p>
            <p style="margin-top: 16px;">Thank you for using AI-as-a-Service!</p>
        </body>
        </html>
        """
        msg.attach(MIMEText(body, "html"))

        with open(report_path, "rb") as f:
            part = MIMEApplication(f.read(), Name=report_path.name)
            part["Content-Disposition"] = f'attachment; filename="{report_path.name}"'
            msg.attach(part)

        return self._send(msg)
