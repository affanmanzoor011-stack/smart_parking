"""
rules.py  -  violation rule + message text + EMAIL sending (Brevo HTTPS API).

Email is sent through Brevo's HTTPS API (not Gmail SMTP).
Uses only Python's built-in urllib (no extra packages).

Environment variables (never store these in code):
    BREVO_API_KEY      your Brevo API key (starts with xkeysib-)
    EMAIL_FROM         a sender email VERIFIED in Brevo
    EMAIL_FROM_NAME    (optional) display name, default "Smart Parking Demo"
If the keys are missing, sending is simulated so the app still works.
"""

import json
import os
import re
import urllib.error
import urllib.request

# zone -> fine amount in Rs (0 = legal area, no violation)
ZONES = {
    "No Parking Zone": 500,
    "Footpath / Pedestrian Area": 1000,
    "Blocking Gate / Driveway": 750,
    "Reserved / Disabled Spot": 1500,
    "Legal Parking Area": 0,
}

SUBJECT = "Parking violation challan (DEMO)"


def check_violation(zone, vehicle_present):
    """Return (is_violation, fine, reason)."""
    fine = ZONES.get(zone, 0)
    if not vehicle_present:
        return False, 0, "No vehicle found in the image."
    if fine == 0:
        return False, 0, f"Vehicle is in '{zone}'. No violation."
    return True, fine, f"Vehicle parked in '{zone}'."


def build_message(challan):
    return (
        f"[DEMO] Challan #{challan['id']}: Vehicle {challan['plate']} was parked in "
        f"{challan['zone']} on {challan['timestamp']}. Fine: Rs {challan['fine']}. "
        f"This is a sample message from a college project, not a real challan."
    )


def valid_email(email):
    return bool(re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", (email or "").strip()))


def email_configured():
    return bool(os.environ.get("BREVO_API_KEY") and os.environ.get("EMAIL_FROM"))


def send_message(to_email, text, subject=SUBJECT):
    """
    Send an email. Always returns a dict:
      {"to", "text", "mode": "email"|"simulated", "ok": bool, "error": str, "id": str}
    """
    to_email = (to_email or "").strip()
    result = {"to": to_email, "text": text, "mode": "simulated", "ok": True,
              "error": "", "id": ""}

    if not email_configured():
        return result
    if not valid_email(to_email):
        result.update(mode="email", ok=False, error="Owner has no valid email address.")
        return result

    payload = {
        "sender": {
            "name": os.environ.get("EMAIL_FROM_NAME", "Smart Parking Demo"),
            "email": os.environ["EMAIL_FROM"].strip(),
        },
        "to": [{"email": to_email}],
        "subject": subject,
        "textContent": text,
    }
    req = urllib.request.Request(
        "https://api.brevo.com/v3/smtp/email",
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "api-key": os.environ["BREVO_API_KEY"].strip(),
            "content-type": "application/json",
            "accept": "application/json",
            "user-agent": "smart-parking-demo/1.0",
        },
    )

    result["mode"] = "email"
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            body = json.loads(resp.read().decode() or "{}")
            result["id"] = body.get("messageId", "")
    except urllib.error.HTTPError as e:
        try:
            msg = json.loads(e.read().decode()).get("message", "")
        except Exception:
            msg = ""
        result.update(ok=False, error=f"Brevo error {e.code}: {msg}")
    except Exception as e:
        result.update(ok=False, error=f"Could not reach Brevo: {e}")
    return result
