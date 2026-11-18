"""
app.py  -  Streamlit GUI (simple, low-memory version)
Upload photo -> detect vehicle + plate -> check zone rule -> approve ->
create challan -> EMAIL via Brevo (or simulated if no keys). Also shows
history and owners.

Environment variables (set in Render):
    BREVO_API_KEY, EMAIL_FROM   -> real email
    APP_PASSWORD                -> login password (optional)

Run:
    py -3.12 -m streamlit run app.py
"""

import csv
import hmac
import io
import math
import os
import re
import tempfile

import streamlit as st

import database as db
from detect_lite import detect_vehicles, detect_plate, save_annotated
from rules import (ZONES, check_violation, build_message, send_message,
                   email_configured, valid_email)

db.init_db()

st.set_page_config(page_title="Smart Parking Violation System", layout="centered")


# ------------------------------------------------------------------ helpers
def valid_phone(phone):
    """Phone is optional. If given: +91XXXXXXXXXX style or 10-13 digits."""
    phone = (phone or "").replace(" ", "")
    return phone == "" or bool(re.fullmatch(r"\+?\d{10,13}", phone))


def md_table(headers, rows):
    """Small markdown table (so we do not need pandas)."""
    esc = lambda v: str(v).replace("|", "/").replace("\n", " ")
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(esc(v) for v in r) + " |")
    return "\n".join(lines)


def remove_files(*paths):
    for p in paths:
        try:
            if p and os.path.exists(p):
                os.remove(p)
        except OSError:
            pass


def status_of(sent):
    """Challan status text from the email result."""
    if not sent["ok"]:
        return "Failed"
    return "Sent" if sent["mode"] == "email" else "Simulated (no email)"


# -------------------------------------------------------------------- login
def password_gate():
    """If APP_PASSWORD is set, ask for it before showing anything."""
    expected = os.environ.get("APP_PASSWORD", "")
    if not expected or st.session_state.get("auth"):
        return
    st.title("Smart Parking Violation System")
    pw = st.text_input("Password", type="password")
    if st.button("Login", type="primary"):
        if hmac.compare_digest(pw.encode(), expected.encode()):
            st.session_state["auth"] = True
            st.rerun()
        else:
            st.error("Wrong password.")
    st.stop()


password_gate()

# ------------------------------------------------------------------- header
st.title("Smart Parking Violation System")
st.caption("DEMO project. No real challans are issued and no real owner data is used.")
if not email_configured():
    st.caption("Email mode: SIMULATED (Brevo keys not set).")

page = st.radio(
    "Menu", ["Check vehicle", "Challan history", "Owners (dummy)"],
    horizontal=True, label_visibility="collapsed",
)

# ------------------------------------------------------------ Check vehicle
if page == "Check vehicle":
    zone = st.selectbox("Location / zone", list(ZONES.keys()))
    uploaded = st.file_uploader("Upload vehicle image", type=["jpg", "jpeg", "png"])

    if uploaded is None:
        st.info("1. Pick the zone   2. Upload a photo   3. Check the plate   4. Approve the challan")
    else:
        key = f"{uploaded.name}-{uploaded.size}"

        # Analyse only once per image (not on every button click)
        if st.session_state.get("img_key") != key:
            remove_files(st.session_state.get("img_path"), st.session_state.get("annotated"))
            suffix = os.path.splitext(uploaded.name)[1] or ".jpg"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.getbuffer())
                img_path = tmp.name
            with st.spinner("Analysing image (can take a while on the free server)..."):
                vehicles = detect_vehicles(img_path)
                plate = detect_plate(img_path)
                annotated = save_annotated(img_path, img_path + "_out.jpg")
            st.session_state.update(
                img_key=key, img_path=img_path, vehicles=vehicles,
                plate=plate, annotated=annotated,
            )

        s = st.session_state
        st.image(s["annotated"], caption="Detection result", width="stretch")

        if s["vehicles"]:
            st.success(
                f"{len(s['vehicles'])} vehicle(s) detected: "
                + ", ".join(f"{v['type']} ({v['confidence']})" for v in s["vehicles"])
            )
        else:
            st.warning("No full vehicle found (close-up photo?). Reading the plate from the whole image.")

        plate_text = st.text_input(
            "Number plate (edit if the reading is wrong)", value=s["plate"], key=f"plate_{key}"
        )
        plate_clean = db.normalize(plate_text)

        is_violation, fine, reason = check_violation(
            zone, bool(s["vehicles"]) or bool(plate_clean)
        )

        if not is_violation:
            st.success(f"No violation. {reason}")
        else:
            st.error(f"VIOLATION: {reason}  |  Fine: Rs {fine}")

            owner = db.get_owner(plate_clean)
            has_email = bool(owner and valid_email(owner.get("email", "")))
            if has_email:
                st.write(f"Owner: **{owner['name']}**  |  Email: **{owner['email']}**")
            elif plate_clean:
                if owner:
                    st.warning(f"Owner '{owner['name']}' has no email. Add it below to continue.")
                else:
                    st.warning("Owner not found in the database. Add the owner below to continue.")
                with st.form(f"quick_owner_{plate_clean}"):
                    q_name = st.text_input("Owner name", value=owner["name"] if owner else "")
                    q_email = st.text_input("Owner email")
                    q_phone = st.text_input("Phone (optional)", value=owner["phone"] if owner else "")
                    if st.form_submit_button("Save owner"):
                        if not q_name.strip():
                            st.error("Please enter the owner name.")
                        elif not valid_email(q_email):
                            st.error("Enter a valid email address.")
                        elif not valid_phone(q_phone):
                            st.error("Enter a valid phone number (10 to 13 digits) or leave it empty.")
                        else:
                            db.add_owner(plate_clean, q_name, q_phone, q_email)
                            st.rerun()

            # Was a challan already created for this image + plate + zone?
            done = st.session_state.get("done")
            already = bool(
                done and done["key"] == key and done["zone"] == zone
                and done["challan"]["plate"] == plate_clean
            )

            # Same plate + zone challaned in the last hour?
            recent = None if already or not plate_clean else db.find_recent_challan(plate_clean, zone)
            override = False
            if recent:
                st.warning(
                    f"Challan #{recent['id']} for this plate in this zone was already "
                    f"created at {recent['timestamp']}."
                )
                override = st.checkbox("Send another challan anyway", key=f"ovr_{key}")

            # Human review step: nothing is sent until you approve
            clicked = st.button(
                "Challan already sent" if already else "Approve & send challan",
                type="primary",
                disabled=(not has_email or not plate_clean or already
                          or (recent is not None and not override)),
            )
            if clicked:
                challan = db.create_challan(
                    plate_clean, owner, zone, fine, s["img_path"], status="Pending"
                )
                with st.spinner("Sending email..."):
                    sent = send_message(owner["email"], build_message(challan))
                challan["status"] = status_of(sent)
                db.update_challan_status(challan["id"], challan["status"])
                st.session_state["done"] = {
                    "key": key, "zone": zone, "challan": challan, "sent": sent,
                }
                done, already = st.session_state["done"], True

            if already:
                c, sent = done["challan"], done["sent"]
                if sent["ok"]:
                    if sent["mode"] == "email":
                        st.success(f"Challan #{c['id']} created and email sent to {sent['to']}.")
                    else:
                        st.success(f"Challan #{c['id']} created. Email simulated (Brevo keys not set).")
                else:
                    st.error(f"Challan #{c['id']} saved, but the email failed: {sent['error']}")
                    if st.button("Retry sending email"):
                        with st.spinner("Sending email..."):
                            new = send_message(c["email"], build_message(c))
                        db.update_challan_status(c["id"], status_of(new))
                        c["status"] = status_of(new)
                        done["sent"] = new
                        st.rerun()
                st.write(f"Email to {sent['to']} ({sent['mode']}):")
                st.code(sent["text"], language=None)

# ---------------------------------------------------------- Challan history
elif page == "Challan history":
    PAGE_SIZE = 20
    n_all, fines_all = db.challan_totals()
    st.write(
        f"Total challans: **{n_all}**  |  Total fines: **Rs {fines_all}**"
        f"  |  Owners in database: **{db.count_owners()}**"
    )

    if n_all == 0:
        st.info("No challans yet. Create one from the 'Check vehicle' page.")
    else:
        q = st.text_input("Search by plate, owner name or email")
        total = db.count_challans(q)
        pages = max(1, math.ceil(total / PAGE_SIZE))
        page_no = 1
        if pages > 1:
            page_no = st.number_input(f"Page (1 to {pages})", min_value=1, max_value=pages,
                                      value=1, step=1, key=f"pg_{q}")
        rows = db.search_challans(q, PAGE_SIZE, (int(page_no) - 1) * PAGE_SIZE)
        st.caption(f"{total} matching challan(s). Showing {len(rows)} on this page, newest first.")

        headers = ["ID", "Plate", "Owner", "Email", "Zone", "Fine (Rs)", "Time", "Status"]
        to_row = lambda r: [r["id"], r["plate"], r["owner_name"], r["email"], r["zone"],
                            r["fine"], r["timestamp"], r["status"]]

        if not rows:
            st.warning("No challans match your search.")
        else:
            st.markdown(md_table(headers, [to_row(r) for r in rows]))

            # CSV of everything that matches the search (not just this page)
            buf = io.StringIO()
            w = csv.writer(buf)
            w.writerow(headers)
            for r in db.search_challans(q, 100000, 0):
                w.writerow(to_row(r))
            st.download_button("Download matching challans (CSV)", buf.getvalue().encode("utf-8"),
                               file_name="challan_history.csv", mime="text/csv")

            st.subheader("View a challan")
            options = [f"#{r['id']} - {r['plate']} - {r['timestamp']}" for r in rows]
            r = rows[options.index(st.selectbox("Select challan (from this page)", options))]
            if os.path.exists(r["image_path"]):
                st.image(r["image_path"], caption="Evidence image", width="stretch")
            st.write(f"**{r['plate']}**  |  {r['zone']}  |  Rs {r['fine']}  |  {r['owner_name']}  |  {r['email']}  |  {r['status']}")
            st.code(build_message(r), language=None)
            if st.button("Delete this challan", key=f"del_ch_{r['id']}"):
                db.delete_challan(r["id"])
                st.rerun()

    # clear all challans
    if n_all:
        with st.expander("Danger zone: delete all challans"):
            sure = st.checkbox("Yes, delete ALL challans and evidence images", key="sure_ch")
            if st.button("Delete all challans", disabled=not sure):
                db.clear_challans()
                st.rerun()

# ------------------------------------------------------------------ Owners
else:
    st.write("Add or update an owner (same plate = overwrite). Use fake or your own details only.")
    with st.form("add_owner", clear_on_submit=True):
        new_plate = st.text_input("Plate number")
        new_name = st.text_input("Owner name")
        new_email = st.text_input("Owner email")
        new_phone = st.text_input("Phone (optional)")
        if st.form_submit_button("Save owner"):
            if not (db.normalize(new_plate) and new_name.strip()):
                st.error("Please fill the plate number and owner name.")
            elif not valid_email(new_email):
                st.error("Enter a valid email address.")
            elif not valid_phone(new_phone):
                st.error("Enter a valid phone number (10 to 13 digits) or leave it empty.")
            else:
                db.add_owner(new_plate, new_name, new_phone, new_email)
                st.success("Owner saved.")

    st.subheader("Find an owner")
    oq = st.text_input("Search owners (plate, name or email)")
    total_o, owners = db.search_owners(oq, limit=50)
    if owners:
        st.caption(f"{total_o} owner(s) match. Showing the first {len(owners)}.")
        st.markdown(md_table(["Plate", "Owner", "Email", "Phone"],
                             [[o["plate"], o["name"], o["email"] or "-", o["phone"] or "-"] for o in owners]))
    else:
        st.info("No owners found.")

    if owners:
        pick = st.selectbox("Delete an owner", [f"{o['plate']} - {o['name']}" for o in owners],
                            key="del_owner_pick")
        if st.button("Delete selected owner"):
            db.delete_owner(pick.split(" - ")[0])
            st.rerun()

    if db.count_owners():
        with st.expander("Danger zone: delete all owners"):
            sure_o = st.checkbox("Yes, delete ALL owners", key="sure_ow")
            if st.button("Delete all owners", disabled=not sure_o):
                db.clear_owners()
                st.rerun()
