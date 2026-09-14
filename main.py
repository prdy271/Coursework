import os
import shutil
import uuid
import re
import html as html_lib
from datetime import datetime, timedelta
from typing import List
import markdown as md
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse, Response
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from db import get_conn
from ingestion import ingest_resource, ingest_text
from rag import (
    answer_question,
    generate_answers_for_question_source,
    generate_discussion_title,
    detect_export_categories,
    generate_export_document,
    CATEGORY_LABELS,
    exam_intake_reply,
    exam_advisor_reply,
    generate_prep_sheet,
    regenerate_prep_sheet,
    generate_initial_answer_draft,
    regenerate_answer_draft,
    get_resource,
    search_resources_semantic,
    keyword_match_resources,
    generate_topic_tags,
    AIServiceError,
)
from pdf_export import markdown_to_pdf_bytes

load_dotenv()

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.exception_handler(AIServiceError)
async def ai_service_error_handler(request: Request, exc: AIServiceError):
    back_url = request.headers.get("referer") or "/"
    return templates.TemplateResponse(
        request,
        "ai_error.html",
        {"message": str(exc), "back_url": back_url},
        status_code=503,
    )


UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

def filename_search_resources(course_id: int, q: str, rtype: str = None, rfiletype: str = None, limit: int = 20):
    """Fuzzy (typo-tolerant) filename search using pg_trgm similarity."""
    conditions = ["course_id = %s"]
    params = [course_id]
    if rtype:
        conditions.append("type = %s")
        params.append(rtype)
    if rfiletype:
        conditions.append("filetype = %s")
        params.append(rfiletype)
    where_clause = " AND ".join(conditions)

    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        f"""
        SELECT id, name, type, subtype, question_category, filetype, similarity(name, %s) AS sim
        FROM resources
        WHERE {where_clause} AND similarity(name, %s) > 0.1
        ORDER BY sim DESC
        LIMIT %s
        """,
        [q] + params + [q, limit],
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


def _result_item(r):
    rid, name, rtype_, subtype, qcat, filetype = r[0], r[1], r[2], r[3], r[4], r[5]
    tag_parts = [rtype_.replace("_", " ")]
    if subtype:
        tag_parts.append(subtype.replace("_", " "))
    if qcat:
        tag_parts.append(qcat.replace("_", " "))
    return {"id": rid, "name": name, "filetype": filetype, "tag": " \u00b7 ".join(tag_parts)}


def _bucket_search_results(rows):
    """rows: (id, name, type, subtype, question_category, filetype, score) -- best first."""
    items = [_result_item(r) for r in rows]
    return {"top": items[:5], "other": items[5:20]}


def _merge_search_results(keyword_rows, semantic_rows):
    """Keyword (full-text) hits are the more certain signal for a specific
    term, so they always land in Top matches. Semantic hits fill any
    remaining Top slots, then the rest go to Other."""
    top_items = []
    seen_ids = set()
    for r in keyword_rows:
        top_items.append(_result_item(r))
        seen_ids.add(r[0])

    other_items = []
    for r in semantic_rows:
        if r[0] in seen_ids:
            continue
        seen_ids.add(r[0])
        item = _result_item(r)
        if len(top_items) < 5:
            top_items.append(item)
        else:
            other_items.append(item)

    return {"top": top_items, "other": other_items[:15]}


def topic_match_resources(course_id: int, query: str, rtype: str = None, rfiletype: str = None, limit: int = 20):
    """Resources whose topic tags mention the query -- a strong signal since
    a tag reflects the model actually reading and understanding the content,
    not just literal text matching."""
    conditions = ["course_id = %s", "topics IS NOT NULL"]
    params = [course_id]
    if rtype:
        conditions.append("type = %s")
        params.append(rtype)
    if rfiletype:
        conditions.append("filetype = %s")
        params.append(rfiletype)
    where_clause = " AND ".join(conditions)

    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        f"""
        SELECT id, name, type, subtype, question_category, filetype
        FROM resources
        WHERE {where_clause} AND EXISTS (SELECT 1 FROM unnest(topics) t WHERE t ILIKE %s)
        LIMIT %s
        """,
        params + [f"%{query.lower()}%", limit],
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


def tag_resource(resource_id: int, text: str):
    """Assign topic tags to a resource right after it's created (upload,
    export, generated answer set, etc). Best-effort -- a tagging failure
    should never block the resource from being created."""
    if not text or not text.strip():
        return
    tags = generate_topic_tags(text)
    if tags:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("UPDATE resources SET topics=%s WHERE id=%s", (tags, resource_id))
        conn.commit()
        cur.close()
        conn.close()


def get_topic_coverage(course_id: int):
    """How many resources touch each topic, across everything tagged in this
    course. Doesn't require topic-level weightage to be useful -- just
    showing where material is concentrated (or missing) already helps."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT unnest(topics) AS topic, count(*) AS cnt
        FROM resources
        WHERE course_id = %s AND topics IS NOT NULL
        GROUP BY topic
        ORDER BY cnt DESC, topic ASC
        LIMIT 25
        """,
        (course_id,),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [{"topic": t, "count": c} for t, c in rows]


CARD_COLORS = ["violet", "sage", "amber", "rose"]


def render_markdown(text: str) -> str:
    """Escape raw HTML first (so a message can never inject a live tag), then
    render markdown syntax on top -- safe to output with |safe in templates.

    LaTeX math ($...$ and $$...$$) is pulled out before markdown runs and
    restored verbatim afterward -- markdown's own syntax (e.g. underscores
    for italics) would otherwise mangle math like x_1 or \\frac{a}{b}.
    KaTeX (loaded in _chat_script.html) renders the math client-side."""
    escaped = html_lib.escape(text or "")

    math_blocks = []

    def _stash(m):
        math_blocks.append(m.group(0))
        return f"@@MATH{len(math_blocks) - 1}@@"

    protected = re.sub(r"\$\$.*?\$\$|\$[^\$\n]+?\$", _stash, escaped, flags=re.DOTALL)
    html = md.markdown(protected, extensions=["fenced_code", "tables", "nl2br"])

    for i, block in enumerate(math_blocks):
        html = html.replace(f"@@MATH{i}@@", block)

    return html


def render_messages(rows):
    """rows: (id, role, content, feedback) -> (id, role, rendered_html, feedback)"""
    return [(mid, role, render_markdown(content), feedback) for mid, role, content, feedback in rows]


def format_weightage_badges(weightage):
    if not isinstance(weightage, dict):
        return []
    badges = []
    for i, (label, pct) in enumerate(weightage.items()):
        badges.append({"label": label, "pct": pct, "color": CARD_COLORS[i % len(CARD_COLORS)]})
    return badges


def format_exam_badges(exam_dates):
    if not isinstance(exam_dates, list):
        return []
    badges = []
    for item in exam_dates:
        name = item.get("name", "Event")
        date_str = item.get("date", "")
        try:
            d = datetime.strptime(date_str, "%Y-%m-%d")
            badges.append({"name": name, "month": d.strftime("%b").upper(), "day": d.strftime("%d")})
        except (ValueError, TypeError):
            badges.append({"name": name, "month": "", "day": date_str})
    return badges


def log_event(course_id: int, name: str, category: str, ref_type: str = None, ref_id: int = None):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO events (course_id, name, category, ref_type, ref_id) VALUES (%s, %s, %s, %s, %s)",
        (course_id, name, category, ref_type, ref_id),
    )
    conn.commit()
    cur.close()
    conn.close()


def get_course(course_id: int):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """SELECT id, name, code, instructor, venue, weightage, grading_policy, exam_dates
           FROM courses WHERE id=%s""",
        (course_id,),
    )
    course = cur.fetchone()
    cur.close()
    conn.close()
    return course


# ---------- Course list / creation ----------

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id, name, code, instructor FROM courses ORDER BY created_at DESC")
    courses = cur.fetchall()
    cur.close()
    conn.close()
    courses_with_color = [
        {"id": c[0], "name": c[1], "code": c[2], "instructor": c[3], "color": CARD_COLORS[i % len(CARD_COLORS)]}
        for i, c in enumerate(courses)
    ]
    return templates.TemplateResponse(request, "index.html", {"courses": courses_with_color})


@app.get("/courses/new", response_class=HTMLResponse)
def new_course_form(request: Request):
    return templates.TemplateResponse(request, "new_course.html", {})


@app.post("/courses")
def create_course(
    name: str = Form(...),
    code: str = Form(""),
    instructor: str = Form(""),
    venue: str = Form(""),
    weightage: str = Form("{}"),
    grading_policy: str = Form(""),
    exam_dates: str = Form("[]"),
):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO courses (name, code, instructor, venue, weightage, grading_policy, exam_dates)
           VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id""",
        (name, code, instructor, venue, weightage, grading_policy, exam_dates),
    )
    conn.commit()
    cur.close()
    conn.close()
    return RedirectResponse("/", status_code=303)


# ---------- Course home (dashboard/nav) ----------

@app.get("/courses/{course_id}", response_class=HTMLResponse)
def course_home(request: Request, course_id: int):
    course = get_course(course_id)
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM resources WHERE course_id=%s", (course_id,))
    resource_count = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM discussions WHERE course_id=%s", (course_id,))
    discussion_count = cur.fetchone()[0]
    cur.close()
    conn.close()
    return templates.TemplateResponse(
        request,
        "course_home.html",
        {
            "course": course,
            "resource_count": resource_count,
            "discussion_count": discussion_count,
            "weightage_badges": format_weightage_badges(course[5]),
            "exam_badges": format_exam_badges(course[7]),
            "topic_coverage": get_topic_coverage(course_id),
        },
    )


# ---------- Resource tree ----------

@app.get("/courses/{course_id}/resources", response_class=HTMLResponse)
def resources_view(request: Request, course_id: int, q: str = "", mode: str = "filename", rtype: str = "", rfiletype: str = ""):
    course = get_course(course_id)
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """SELECT id, type, subtype, question_category, name, filetype, linked_question_source_id
           FROM resources WHERE course_id=%s ORDER BY created_at DESC""",
        (course_id,),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()

    tree = {"book": {}, "note": {}, "question_source": {}, "answer_set": []}
    for r in rows:
        rid, rtype_, subtype, qcat, name, filetype, linked_qs = r
        item = {"id": rid, "name": name, "filetype": filetype, "linked_question_source_id": linked_qs}
        if rtype_ == "answer_set":
            tree["answer_set"].append(item)
        elif rtype_ == "question_source":
            key = qcat or "uncategorized"
            tree["question_source"].setdefault(key, []).append(item)
        elif rtype_ in ("book", "note"):
            key = subtype or "uncategorized"
            tree[rtype_].setdefault(key, []).append(item)

    search_results = None
    if q.strip():
        if mode == "semantic":
            semantic_rows = search_resources_semantic(course_id, q, rtype or None, rfiletype or None)
            keyword_rows = keyword_match_resources(course_id, q, rtype or None, rfiletype or None)
            topic_rows = topic_match_resources(course_id, q, rtype or None, rfiletype or None)
            search_results = _merge_search_results(keyword_rows + topic_rows, semantic_rows)
        else:
            raw = filename_search_resources(course_id, q, rtype or None, rfiletype or None)
            search_results = _bucket_search_results(raw)

    return templates.TemplateResponse(
        request,
        "resources.html",
        {"course": course, "tree": tree, "q": q, "mode": mode, "rtype": rtype, "rfiletype": rfiletype, "search_results": search_results},
    )


@app.post("/courses/{course_id}/resources")
async def upload_resource(
    course_id: int,
    type: str = Form(...),
    subtype: str = Form(""),
    question_category: str = Form(""),
    files: List[UploadFile] = File(...),
):
    for file in files:
        ext = file.filename.split(".")[-1].lower()
        saved_name = f"{uuid.uuid4()}.{ext}"
        path = os.path.join(UPLOAD_DIR, saved_name)
        with open(path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        conn = get_conn()
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO resources (course_id, type, subtype, question_category, name, filetype, file_size, file_path)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING id""",
            (course_id, type, subtype or None, question_category or None, file.filename, ext, os.path.getsize(path), saved_name),
        )
        resource_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
        conn.close()

        ingest_resource(resource_id, path, ext)

        conn = get_conn()
        cur = conn.cursor()
        cur.execute("SELECT raw_text FROM resources WHERE id=%s", (resource_id,))
        raw_text_row = cur.fetchone()
        cur.close()
        conn.close()
        if raw_text_row and raw_text_row[0]:
            tag_resource(resource_id, raw_text_row[0])

        log_event(course_id, f"Uploaded {file.filename}", "data_upload", ref_type="resource", ref_id=resource_id)

    return RedirectResponse(f"/courses/{course_id}/resources", status_code=303)


@app.get("/resources/{resource_id}/download")
def download_resource(resource_id: int):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT name, filetype, file_path, raw_text FROM resources WHERE id=%s", (resource_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()

    if not row:
        return Response("Not found", status_code=404)
    name, filetype, file_path, raw_text = row

    def with_extension(base_name: str, ext: str) -> str:
        ext = (ext or "").lower()
        if ext and not base_name.lower().endswith(f".{ext}"):
            return f"{base_name}.{ext}"
        return base_name

    if file_path:
        full_path = os.path.join(UPLOAD_DIR, file_path)
        if os.path.exists(full_path):
            return FileResponse(full_path, filename=with_extension(name, filetype))

    # No physical file (e.g. a generated answer set) -- serve the text content directly.
    return Response(
        content=raw_text or "",
        media_type="text/plain",
        headers={"Content-Disposition": f'attachment; filename="{with_extension(name, filetype or "txt")}"'},
    )


@app.post("/resources/{resource_id}/delete")
def delete_resource(resource_id: int):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT course_id, file_path FROM resources WHERE id=%s", (resource_id,))
    row = cur.fetchone()
    if not row:
        cur.close()
        conn.close()
        return RedirectResponse("/", status_code=303)
    course_id, file_path = row

    cur.execute("DELETE FROM resources WHERE id=%s", (resource_id,))
    conn.commit()
    cur.close()
    conn.close()

    if file_path:
        full_path = os.path.join(UPLOAD_DIR, file_path)
        if os.path.exists(full_path):
            try:
                os.remove(full_path)
            except OSError:
                pass

    return RedirectResponse(f"/courses/{course_id}/resources", status_code=303)


@app.get("/resources/{resource_id}/edit", response_class=HTMLResponse)
def edit_resource_form(request: Request, resource_id: int):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT course_id, name, type, subtype, question_category FROM resources WHERE id=%s", (resource_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    if not row:
        return RedirectResponse("/", status_code=303)
    course_id, name, rtype, subtype, qcat = row
    course = get_course(course_id)
    return templates.TemplateResponse(
        request,
        "edit_resource.html",
        {"course": course, "resource_id": resource_id, "name": name, "rtype": rtype, "subtype": subtype, "qcat": qcat},
    )


@app.post("/resources/{resource_id}/edit")
def edit_resource_save(resource_id: int, name: str = Form(...), subtype: str = Form(""), question_category: str = Form("")):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT course_id FROM resources WHERE id=%s", (resource_id,))
    row = cur.fetchone()
    if not row:
        cur.close()
        conn.close()
        return RedirectResponse("/", status_code=303)
    course_id = row[0]
    cur.execute(
        "UPDATE resources SET name=%s, subtype=%s, question_category=%s WHERE id=%s",
        (name, subtype or None, question_category or None, resource_id),
    )
    conn.commit()
    cur.close()
    conn.close()
    return RedirectResponse(f"/courses/{course_id}/resources", status_code=303)


# ---------- Discussions (ChatGPT/Claude-style split layout) ----------

def _get_discussion_sidebar(course_id: int):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, title, updated_at FROM discussions WHERE course_id=%s AND mode='general' ORDER BY updated_at DESC",
        (course_id,),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


@app.get("/courses/{course_id}/discussions", response_class=HTMLResponse)
def discussions_new(request: Request, course_id: int):
    course = get_course(course_id)
    discussions = _get_discussion_sidebar(course_id)
    return templates.TemplateResponse(
        request,
        "discussions.html",
        {"course": course, "discussions": discussions, "active_discussion_id": None, "messages": []},
    )


@app.get("/courses/{course_id}/discussions/{discussion_id}", response_class=HTMLResponse)
def discussions_view(request: Request, course_id: int, discussion_id: int):
    course = get_course(course_id)
    discussions = _get_discussion_sidebar(course_id)
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, role, content, feedback FROM messages WHERE discussion_id=%s ORDER BY created_at ASC",
        (discussion_id,),
    )
    messages = render_messages(cur.fetchall())
    cur.close()
    conn.close()
    return templates.TemplateResponse(
        request,
        "discussions.html",
        {"course": course, "discussions": discussions, "active_discussion_id": discussion_id, "messages": messages},
    )


@app.post("/courses/{course_id}/discussions")
def start_discussion(course_id: int, message: str = Form(...)):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO discussions (course_id, title) VALUES (%s, 'New discussion') RETURNING id",
        (course_id,),
    )
    discussion_id = cur.fetchone()[0]
    cur.execute(
        "INSERT INTO messages (discussion_id, role, content) VALUES (%s, 'user', %s)",
        (discussion_id, message),
    )
    conn.commit()
    cur.close()
    conn.close()

    answer = answer_question(course_id, message, [])

    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO messages (discussion_id, role, content) VALUES (%s, 'assistant', %s)",
        (discussion_id, answer),
    )
    title = generate_discussion_title(message, answer)
    cur.execute("UPDATE discussions SET title=%s, updated_at=now() WHERE id=%s", (title, discussion_id))
    conn.commit()
    cur.close()
    conn.close()

    log_event(course_id, f'Started discussion "{title}"', "discussion", ref_type="discussion", ref_id=discussion_id)
    return RedirectResponse(f"/courses/{course_id}/discussions/{discussion_id}", status_code=303)


@app.post("/courses/{course_id}/discussions/{discussion_id}/chat")
async def chat(course_id: int, discussion_id: int, message: str = Form(...)):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT role, content FROM messages WHERE discussion_id=%s ORDER BY created_at ASC",
        (discussion_id,),
    )
    history = [{"role": r, "content": c} for r, c in cur.fetchall()]

    cur.execute(
        "INSERT INTO messages (discussion_id, role, content) VALUES (%s, 'user', %s)",
        (discussion_id, message),
    )
    conn.commit()

    answer = answer_question(course_id, message, history)

    cur.execute(
        "INSERT INTO messages (discussion_id, role, content) VALUES (%s, 'assistant', %s)",
        (discussion_id, answer),
    )
    cur.execute("UPDATE discussions SET updated_at = now() WHERE id=%s", (discussion_id,))
    conn.commit()
    cur.close()
    conn.close()

    log_event(course_id, "New message in discussion", "discussion", ref_type="discussion", ref_id=discussion_id)
    return RedirectResponse(f"/courses/{course_id}/discussions/{discussion_id}", status_code=303)


# ---------- Discussion export (doubts / teaching / plan -> PDF documents) ----------

@app.get("/courses/{course_id}/discussions/{discussion_id}/export", response_class=HTMLResponse)
def export_form(request: Request, course_id: int, discussion_id: int):
    course = get_course(course_id)
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT title FROM discussions WHERE id=%s", (discussion_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    discussion_title = row[0] if row else "Discussion"

    found_categories = detect_export_categories(discussion_id)
    categories = [{"key": c, "label": CATEGORY_LABELS[c]} for c in found_categories]

    return templates.TemplateResponse(
        request,
        "export.html",
        {
            "course": course,
            "discussion_id": discussion_id,
            "discussion_title": discussion_title,
            "categories": categories,
        },
    )


@app.post("/courses/{course_id}/discussions/{discussion_id}/export")
def export_generate(
    course_id: int,
    discussion_id: int,
    categories: List[str] = Form(...),
    mode: str = Form("separate"),
):
    selected = [c for c in categories if c in CATEGORY_LABELS]
    if not selected:
        return RedirectResponse(f"/courses/{course_id}/discussions/{discussion_id}/export", status_code=303)

    category_groups = [[c] for c in selected] if mode == "separate" else [selected]

    last_resource_id = None
    for group in category_groups:
        title, body = generate_export_document(discussion_id, group)
        pdf_bytes = markdown_to_pdf_bytes(title, body)

        saved_name = f"{uuid.uuid4()}.pdf"
        with open(os.path.join(UPLOAD_DIR, saved_name), "wb") as f:
            f.write(pdf_bytes)

        conn = get_conn()
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO resources (course_id, type, subtype, name, filetype, file_size, file_path)
               VALUES (%s, 'note', 'from_discussion', %s, 'pdf', %s, %s) RETURNING id""",
            (course_id, title, len(pdf_bytes), saved_name),
        )
        resource_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
        conn.close()

        ingest_text(resource_id, body)
        tag_resource(resource_id, body)
        log_event(course_id, f'Exported "{title}" from discussion', "data_upload", ref_type="resource", ref_id=resource_id)
        last_resource_id = resource_id

    return RedirectResponse(f"/courses/{course_id}/resources", status_code=303)


# ---------- Generate Answers (session-based: chat + live draft + finish) ----------

def _get_answer_sessions(course_id: int):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """SELECT d.id, d.title, d.updated_at
           FROM discussions d WHERE d.course_id=%s AND d.mode='answer_gen'
           ORDER BY d.updated_at DESC""",
        (course_id,),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


@app.get("/courses/{course_id}/answer", response_class=HTMLResponse)
def answer_new(request: Request, course_id: int):
    course = get_course(course_id)
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, name, question_category FROM resources WHERE course_id=%s AND type='question_source' ORDER BY created_at DESC",
        (course_id,),
    )
    question_sources = cur.fetchall()
    cur.close()
    conn.close()
    sessions = _get_answer_sessions(course_id)
    return templates.TemplateResponse(
        request,
        "answer.html",
        {
            "course": course,
            "question_sources": question_sources,
            "sessions": sessions,
            "active_discussion_id": None,
            "messages": [],
        },
    )


@app.get("/courses/{course_id}/answer/{discussion_id}", response_class=HTMLResponse)
def answer_view(request: Request, course_id: int, discussion_id: int):
    course = get_course(course_id)
    sessions = _get_answer_sessions(course_id)
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, role, content, feedback FROM messages WHERE discussion_id=%s ORDER BY created_at ASC",
        (discussion_id,),
    )
    messages = render_messages(cur.fetchall())
    cur.close()
    conn.close()
    return templates.TemplateResponse(
        request,
        "answer.html",
        {
            "course": course,
            "question_sources": [],
            "sessions": sessions,
            "active_discussion_id": discussion_id,
            "messages": messages,
        },
    )


@app.post("/courses/{course_id}/answer")
def answer_start(course_id: int, question_source_id: int = Form(...)):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT name FROM resources WHERE id=%s", (question_source_id,))
    source_name = cur.fetchone()[0]
    cur.execute(
        """INSERT INTO discussions (course_id, title, mode, target_question_source_id)
           VALUES (%s, %s, 'answer_gen', %s) RETURNING id""",
        (course_id, f"Answers - {source_name}", question_source_id),
    )
    discussion_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()

    reply, draft = generate_initial_answer_draft(course_id, question_source_id)

    conn = get_conn()
    cur = conn.cursor()
    cur.execute("INSERT INTO messages (discussion_id, role, content) VALUES (%s, 'assistant', %s)", (discussion_id, reply))
    cur.execute("UPDATE discussions SET draft_content=%s, updated_at=now() WHERE id=%s", (draft, discussion_id))
    conn.commit()
    cur.close()
    conn.close()

    log_event(course_id, f"Started answer generation for {source_name}", "answer_generation", ref_type="discussion", ref_id=discussion_id)
    return RedirectResponse(f"/courses/{course_id}/answer/{discussion_id}", status_code=303)


@app.post("/courses/{course_id}/answer/{discussion_id}/chat")
def answer_chat(course_id: int, discussion_id: int, message: str = Form(...)):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT role, content FROM messages WHERE discussion_id=%s ORDER BY created_at ASC", (discussion_id,))
    history = [{"role": r, "content": c} for r, c in cur.fetchall()]
    cur.execute("SELECT target_question_source_id, draft_content FROM discussions WHERE id=%s", (discussion_id,))
    question_source_id, draft_content = cur.fetchone()
    cur.execute("INSERT INTO messages (discussion_id, role, content) VALUES (%s, 'user', %s)", (discussion_id, message))
    conn.commit()
    cur.close()
    conn.close()

    reply, new_draft = regenerate_answer_draft(course_id, question_source_id, draft_content or "", message, history)

    conn = get_conn()
    cur = conn.cursor()
    cur.execute("INSERT INTO messages (discussion_id, role, content) VALUES (%s, 'assistant', %s)", (discussion_id, reply))
    cur.execute("UPDATE discussions SET draft_content=%s, updated_at=now() WHERE id=%s", (new_draft, discussion_id))
    conn.commit()
    cur.close()
    conn.close()

    return RedirectResponse(f"/courses/{course_id}/answer/{discussion_id}", status_code=303)


@app.get("/courses/{course_id}/answer/{discussion_id}/preview.pdf")
def answer_preview_pdf(course_id: int, discussion_id: int):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """SELECT d.draft_content, r.name FROM discussions d
           LEFT JOIN resources r ON d.target_question_source_id = r.id
           WHERE d.id=%s""",
        (discussion_id,),
    )
    row = cur.fetchone()
    cur.close()
    conn.close()

    if not row or not row[0]:
        return Response("No draft yet", status_code=404)
    draft_content, source_name = row
    pdf_bytes = markdown_to_pdf_bytes(f"Answers - {source_name or 'Draft'}", draft_content)
    return Response(content=pdf_bytes, media_type="application/pdf")


@app.post("/courses/{course_id}/answer/{discussion_id}/finish")
def answer_finish(course_id: int, discussion_id: int):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT target_question_source_id, draft_content FROM discussions WHERE id=%s", (discussion_id,))
    question_source_id, draft_content = cur.fetchone()
    cur.execute("SELECT name FROM resources WHERE id=%s", (question_source_id,))
    source_name = cur.fetchone()[0]
    cur.close()
    conn.close()

    title = f"Answers - {source_name}"
    pdf_bytes = markdown_to_pdf_bytes(title, draft_content or "")
    saved_name = f"{uuid.uuid4()}.pdf"
    with open(os.path.join(UPLOAD_DIR, saved_name), "wb") as f:
        f.write(pdf_bytes)

    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO resources (course_id, type, subtype, name, filetype, file_size, file_path, linked_question_source_id)
           VALUES (%s, 'answer_set', 'model_generated', %s, 'pdf', %s, %s, %s) RETURNING id""",
        (course_id, title, len(pdf_bytes), saved_name, question_source_id),
    )
    new_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()

    ingest_text(new_id, draft_content or "")
    tag_resource(new_id, draft_content or "")
    log_event(course_id, f"Generated answers for {source_name}", "answer_generation", ref_type="resource", ref_id=new_id)
    return RedirectResponse(f"/courses/{course_id}/resources", status_code=303)


# ---------- Timeline ----------

@app.get("/courses/{course_id}/timeline", response_class=HTMLResponse)
def timeline_view(request: Request, course_id: int, offset: int = 0):
    offset = max(offset, 0)
    course = get_course(course_id)

    today = datetime.now().date()
    window_end = today - timedelta(days=30 * offset)
    window_start = window_end - timedelta(days=29)

    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """SELECT name, category, created_at FROM events
           WHERE course_id=%s AND created_at::date BETWEEN %s AND %s
           ORDER BY created_at ASC""",
        (course_id, window_start, window_end),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()

    def shorten(s: str, limit: int = 70) -> str:
        return s if len(s) <= limit else s[: limit - 1].rstrip() + "\u2026"

    events = [
        {
            "name": shorten(name),
            "full_name": name,
            "category": category,
            "when": created_at.strftime("%b %d, %Y \u00b7 %H:%M"),
        }
        for name, category, created_at in rows
    ]
    return templates.TemplateResponse(
        request,
        "timeline.html",
        {
            "course": course,
            "events": events,
            "offset": offset,
            "window_label": f"{window_start.strftime('%b %d')} \u2013 {window_end.strftime('%b %d, %Y')}",
        },
    )


# ---------- Intent settings ----------

@app.get("/courses/{course_id}/intent", response_class=HTMLResponse)
def intent_form(request: Request, course_id: int):
    course = get_course(course_id)
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT target_grade, depth_preference FROM courses WHERE id=%s", (course_id,))
    target_grade, depth_preference = cur.fetchone()
    cur.close()
    conn.close()
    return templates.TemplateResponse(
        request,
        "intent.html",
        {"course": course, "target_grade": target_grade, "depth_preference": depth_preference or "balanced"},
    )


@app.post("/courses/{course_id}/intent")
def intent_save(course_id: int, has_target: str = Form("no"), target_grade: str = Form(""), depth_preference: str = Form("balanced")):
    grade_value = None
    if has_target == "yes" and target_grade.strip():
        try:
            grade_value = float(target_grade)
        except ValueError:
            grade_value = None

    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "UPDATE courses SET target_grade=%s, depth_preference=%s WHERE id=%s",
        (grade_value, depth_preference, course_id),
    )
    conn.commit()
    cur.close()
    conn.close()
    return RedirectResponse(f"/courses/{course_id}/intent", status_code=303)


# ---------- Exam Mode ----------

def _get_exam_sidebar(course_id: int):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, title, updated_at FROM discussions WHERE course_id=%s AND mode='exam_prep' ORDER BY updated_at DESC",
        (course_id,),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


def _get_tasks(discussion_id: int):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """SELECT id, instruction, linked_resource_id, question_source_id, done
           FROM exam_prep_tasks WHERE discussion_id=%s ORDER BY order_index ASC""",
        (discussion_id,),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()

    tasks = []
    for tid, instruction, linked_resource_id, question_source_id, done in rows:
        resource_name = None
        if linked_resource_id:
            conn2 = get_conn()
            cur2 = conn2.cursor()
            cur2.execute("SELECT name FROM resources WHERE id=%s", (linked_resource_id,))
            r = cur2.fetchone()
            cur2.close()
            conn2.close()
            resource_name = r[0] if r else None

        answer_set_id = None
        if question_source_id:
            conn2 = get_conn()
            cur2 = conn2.cursor()
            cur2.execute(
                "SELECT id FROM resources WHERE linked_question_source_id=%s ORDER BY created_at DESC LIMIT 1",
                (question_source_id,),
            )
            r = cur2.fetchone()
            cur2.close()
            conn2.close()
            answer_set_id = r[0] if r else None

        tasks.append({
            "id": tid, "instruction": instruction, "linked_resource_id": linked_resource_id,
            "resource_name": resource_name, "question_source_id": question_source_id,
            "answer_set_id": answer_set_id, "done": done,
        })
    return tasks


@app.get("/courses/{course_id}/exam-mode", response_class=HTMLResponse)
def exam_mode_new(request: Request, course_id: int):
    course = get_course(course_id)
    sessions = _get_exam_sidebar(course_id)
    return templates.TemplateResponse(
        request,
        "exam_mode.html",
        {"course": course, "sessions": sessions, "active_discussion_id": None, "messages": [], "tasks": []},
    )


@app.get("/courses/{course_id}/exam-mode/{discussion_id}", response_class=HTMLResponse)
def exam_mode_view(request: Request, course_id: int, discussion_id: int):
    course = get_course(course_id)
    sessions = _get_exam_sidebar(course_id)
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, role, content, feedback FROM messages WHERE discussion_id=%s ORDER BY created_at ASC",
        (discussion_id,),
    )
    messages = render_messages(cur.fetchall())
    cur.close()
    conn.close()
    tasks = _get_tasks(discussion_id)
    return templates.TemplateResponse(
        request,
        "exam_mode.html",
        {"course": course, "sessions": sessions, "active_discussion_id": discussion_id, "messages": messages, "tasks": tasks},
    )


@app.post("/courses/{course_id}/exam-mode")
def exam_mode_start(course_id: int, message: str = Form(...)):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO discussions (course_id, title, mode) VALUES (%s, 'New exam prep', 'exam_prep') RETURNING id",
        (course_id,),
    )
    discussion_id = cur.fetchone()[0]
    cur.execute("INSERT INTO messages (discussion_id, role, content) VALUES (%s, 'user', %s)", (discussion_id, message))
    conn.commit()
    cur.close()
    conn.close()

    reply = exam_intake_reply(course_id, message, [])

    conn = get_conn()
    cur = conn.cursor()
    cur.execute("INSERT INTO messages (discussion_id, role, content) VALUES (%s, 'assistant', %s)", (discussion_id, reply))
    title = generate_discussion_title(message, reply)
    cur.execute("UPDATE discussions SET title=%s, updated_at=now() WHERE id=%s", (title, discussion_id))
    conn.commit()
    cur.close()
    conn.close()

    log_event(course_id, f'Started exam prep "{title}"', "discussion", ref_type="discussion", ref_id=discussion_id)
    return RedirectResponse(f"/courses/{course_id}/exam-mode/{discussion_id}", status_code=303)


@app.post("/courses/{course_id}/exam-mode/{discussion_id}/chat")
def exam_mode_chat(course_id: int, discussion_id: int, message: str = Form(...)):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT role, content FROM messages WHERE discussion_id=%s ORDER BY created_at ASC", (discussion_id,))
    history = [{"role": r, "content": c} for r, c in cur.fetchall()]
    cur.execute("INSERT INTO messages (discussion_id, role, content) VALUES (%s, 'user', %s)", (discussion_id, message))
    conn.commit()
    cur.close()
    conn.close()

    tasks = _get_tasks(discussion_id)
    if tasks:
        reply = exam_advisor_reply(course_id, message, history, tasks)
    else:
        reply = exam_intake_reply(course_id, message, history)

    conn = get_conn()
    cur = conn.cursor()
    cur.execute("INSERT INTO messages (discussion_id, role, content) VALUES (%s, 'assistant', %s)", (discussion_id, reply))
    cur.execute("UPDATE discussions SET updated_at=now() WHERE id=%s", (discussion_id,))
    conn.commit()
    cur.close()
    conn.close()

    return RedirectResponse(f"/courses/{course_id}/exam-mode/{discussion_id}", status_code=303)


@app.post("/courses/{course_id}/exam-mode/{discussion_id}/generate")
def exam_mode_generate(course_id: int, discussion_id: int):
    tasks = generate_prep_sheet(course_id, discussion_id)

    conn = get_conn()
    cur = conn.cursor()
    for i, t in enumerate(tasks):
        cur.execute(
            """INSERT INTO exam_prep_tasks (discussion_id, order_index, instruction, linked_resource_id, question_source_id)
               VALUES (%s, %s, %s, %s, %s)""",
            (discussion_id, i, t["instruction"], t["resource_id"], t["question_source_id"]),
        )
    conn.commit()
    cur.close()
    conn.close()

    log_event(course_id, "Generated exam prep sheet", "discussion", ref_type="discussion", ref_id=discussion_id)
    return RedirectResponse(f"/courses/{course_id}/exam-mode/{discussion_id}", status_code=303)


@app.post("/courses/{course_id}/exam-mode/{discussion_id}/update")
def exam_mode_update(course_id: int, discussion_id: int):
    current = _get_tasks(discussion_id)
    current_for_prompt = [{"id": t["id"], "instruction": t["instruction"], "done": t["done"]} for t in current]

    updated = regenerate_prep_sheet(course_id, discussion_id, current_for_prompt)
    kept_ids = {t["id"] for t in updated if t.get("id")}
    done_by_id = {t["id"]: t["done"] for t in current}

    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM exam_prep_tasks WHERE discussion_id=%s AND id != ALL(%s)", (discussion_id, list(kept_ids) or [0]))
    for i, t in enumerate(updated):
        if t.get("id") and t["id"] in done_by_id:
            cur.execute(
                "UPDATE exam_prep_tasks SET order_index=%s, instruction=%s, linked_resource_id=%s, question_source_id=%s WHERE id=%s",
                (i, t["instruction"], t["resource_id"], t["question_source_id"], t["id"]),
            )
        else:
            cur.execute(
                """INSERT INTO exam_prep_tasks (discussion_id, order_index, instruction, linked_resource_id, question_source_id)
                   VALUES (%s, %s, %s, %s, %s)""",
                (discussion_id, i, t["instruction"], t["resource_id"], t["question_source_id"]),
            )
    conn.commit()
    cur.close()
    conn.close()

    log_event(course_id, "Updated exam prep sheet", "discussion", ref_type="discussion", ref_id=discussion_id)
    return RedirectResponse(f"/courses/{course_id}/exam-mode/{discussion_id}", status_code=303)


@app.post("/exam-prep-tasks/{task_id}/toggle")
def toggle_task(task_id: int):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("UPDATE exam_prep_tasks SET done = NOT done WHERE id=%s RETURNING done", (task_id,))
    row = cur.fetchone()
    conn.commit()
    cur.close()
    conn.close()
    return {"done": row[0] if row else None}


@app.post("/messages/{message_id}/feedback")
def message_feedback(message_id: int, value: int = Form(...)):
    """value: 1 (up) or -1 (down). Clicking the same value again clears it."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT feedback FROM messages WHERE id=%s", (message_id,))
    row = cur.fetchone()
    current = row[0] if row else None
    new_value = None if current == value else value
    cur.execute("UPDATE messages SET feedback=%s WHERE id=%s", (new_value, message_id))
    conn.commit()
    cur.close()
    conn.close()
    return {"feedback": new_value}


@app.post("/exam-prep-tasks/{task_id}/generate-answer-key")
def task_generate_answer_key(task_id: int):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """SELECT t.discussion_id, t.question_source_id, d.course_id
           FROM exam_prep_tasks t JOIN discussions d ON t.discussion_id = d.id
           WHERE t.id=%s""",
        (task_id,),
    )
    row = cur.fetchone()
    cur.close()
    conn.close()

    if not row or not row[1]:
        return RedirectResponse("/", status_code=303)
    discussion_id, question_source_id, course_id = row

    content = generate_answers_for_question_source(course_id, question_source_id)

    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT name FROM resources WHERE id=%s", (question_source_id,))
    source_name = cur.fetchone()[0]
    answer_name = f"Answers - {source_name}"
    cur.execute(
        """INSERT INTO resources (course_id, type, subtype, name, filetype, raw_text, linked_question_source_id)
           VALUES (%s, 'answer_set', 'model_generated', %s, 'txt', %s, %s) RETURNING id""",
        (course_id, answer_name, content, question_source_id),
    )
    new_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()

    log_event(course_id, f"Generated answers for {source_name}", "answer_generation", ref_type="resource", ref_id=new_id)
    tag_resource(new_id, content)
    return RedirectResponse(f"/courses/{course_id}/exam-mode/{discussion_id}", status_code=303)
