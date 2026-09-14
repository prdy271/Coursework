import os
import json
import time
from google import genai
from dotenv import load_dotenv
from db import get_conn
from ingestion import get_model

load_dotenv()
_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

GEN_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview")
print(f"[rag.py] Using Gemini model: {GEN_MODEL}", flush=True)

DEPTH_INSTRUCTIONS = {
    "efficient": "The student wants an EFFICIENT path to their target -- focus only on what's directly needed, skip tangents and extra depth.",
    "balanced": "The student wants a BALANCED approach -- cover what's needed plus reasonable context, without going too deep.",
    "deep": "The student wants a DEEP understanding -- go beyond just what's testable, include related concepts and reasoning where it helps real understanding.",
}


class AIServiceError(Exception):
    """Raised when Gemini is unreachable/overloaded after retries -- caught
    at the route level to show a friendly message instead of a raw 500."""
    pass


def _generate(prompt: str, retries: int = 3, base_delay: float = 2.0) -> str:
    """Every Gemini call in this app goes through here. Retries transient
    server-side overload (503 UNAVAILABLE) with backoff before giving up,
    so an occasional busy period doesn't crash the request outright."""
    last_error = None
    for attempt in range(retries):
        try:
            response = _client.models.generate_content(model=GEN_MODEL, contents=prompt)
            return response.text
        except Exception as e:
            last_error = e
            msg = str(e)
            print(f"[_generate] Attempt {attempt + 1}/{retries} failed: {type(e).__name__}: {msg[:200]}", flush=True)
            transient = "503" in msg or "UNAVAILABLE" in msg or "overloaded" in msg.lower()
            if transient and attempt < retries - 1:
                time.sleep(base_delay * (2 ** attempt))
                continue
            break
    print(f"[_generate] Giving up after {retries} attempts. Last error: {type(last_error).__name__}: {last_error}", flush=True)
    raise AIServiceError(
        "The AI service is currently busy or unreachable. This is usually temporary -- please try again in a moment."
    ) from last_error


def retrieve_chunks(course_id: int, query: str, k: int = 6):
    model = get_model()
    query_emb = model.encode([query])[0].tolist()

    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT rc.chunk_text, r.name, r.subtype, rc.page_number
        FROM resource_chunks rc
        JOIN resources r ON rc.resource_id = r.id
        WHERE r.course_id = %s
        ORDER BY rc.embedding <=> %s::vector
        LIMIT %s
        """,
        (course_id, query_emb, k),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


def retrieve_and_rerank(course_id: int, query: str, k_final: int = 6, k_broad: int = 24):
    """Hybrid retrieval (embedding closeness + literal keyword match at the
    chunk level) to get a broad candidate pool, then a cheap re-ranking pass
    to narrow it down to the most relevant few before generation. Returns
    the same (text, name, subtype, page) shape as retrieve_chunks."""
    model = get_model()
    query_emb = model.encode([query])[0].tolist()

    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT rc.id, rc.chunk_text, r.name, r.subtype, rc.page_number
        FROM resource_chunks rc
        JOIN resources r ON rc.resource_id = r.id
        WHERE r.course_id = %s
        ORDER BY rc.embedding <=> %s::vector
        LIMIT %s
        """,
        (course_id, query_emb, k_broad),
    )
    embedding_rows = cur.fetchall()

    cur.execute(
        """
        SELECT rc.id, rc.chunk_text, r.name, r.subtype, rc.page_number
        FROM resource_chunks rc
        JOIN resources r ON rc.resource_id = r.id
        WHERE r.course_id = %s
          AND to_tsvector('english', rc.chunk_text) @@ plainto_tsquery('english', %s)
        LIMIT 10
        """,
        (course_id, query),
    )
    keyword_rows = cur.fetchall()
    cur.close()
    conn.close()

    seen = {}
    for row in keyword_rows + embedding_rows:  # keyword first so a keyword hit is never dropped by dedup
        cid = row[0]
        if cid not in seen:
            seen[cid] = row
    merged = list(seen.values())[:k_broad]

    candidates = [(r[1], r[2], r[3], r[4]) for r in merged]  # drop id -> (text, name, subtype, page)
    if len(candidates) <= k_final:
        return candidates
    return rerank_chunks(query, candidates, k_final)


def rerank_chunks(query: str, candidates: list, k_final: int = 6) -> list:
    """One cheap call: given a broad candidate pool, pick and order the
    genuinely most relevant few. Falls back to the first k_final on any
    parsing failure, so a bad response never breaks retrieval entirely."""
    listing = "\n".join(
        f"[{i}] ({name}{f', page {page}' if page else ''}): {text[:300]}"
        for i, (text, name, subtype, page) in enumerate(candidates)
    )
    prompt = f"""Below is a search query and a numbered list of candidate excerpts from
course material. Return ONLY a JSON array of the {k_final} excerpt numbers most
relevant to the query, ordered best first. If fewer than {k_final} are genuinely
relevant, return only those.

Query: {query}

CANDIDATES:
{listing}
"""
    try:
        response = _generate(prompt)
        indices = json.loads(_strip_json_fence(response))
        picked = [candidates[i] for i in indices if isinstance(i, int) and 0 <= i < len(candidates)]
        return picked[:k_final] if picked else candidates[:k_final]
    except Exception:
        return candidates[:k_final]


def rewrite_query_for_retrieval(message: str, history: list[dict]) -> str:
    """Resolve vague follow-ups ('explain that more') into a self-contained
    search query using conversation history, before embedding/searching.
    The ORIGINAL message still goes to the model for the actual answer --
    this rewritten version is only used to search."""
    if not history:
        return message

    convo = "\n".join(f"{m['role']}: {m['content']}" for m in history[-6:])
    prompt = f"""Given this conversation and a new student message, rewrite the message
into a self-contained search query capturing what they're actually asking
about (resolve pronouns like "that" or "it" using the conversation). Output
ONLY the rewritten query, nothing else.

Conversation:
{convo}

New message: {message}
"""
    try:
        response = _generate(prompt)
        rewritten = response.strip()
        return rewritten if rewritten else message
    except Exception:
        return message


def search_resources_semantic(course_id: int, query: str, rtype: str = None, rfiletype: str = None, limit_chunks: int = 60):
    """Embed the query and search chunk embeddings, then roll chunk-level hits
    up to their parent resource (best/lowest distance wins). Returns rows of
    (id, name, type, subtype, question_category, filetype, distance), best first."""
    model = get_model()
    query_emb = model.encode([query])[0].tolist()

    conditions = ["r.course_id = %s"]
    params = [course_id]
    if rtype:
        conditions.append("r.type = %s")
        params.append(rtype)
    if rfiletype:
        conditions.append("r.filetype = %s")
        params.append(rfiletype)
    where_clause = " AND ".join(conditions)

    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        f"""
        SELECT r.id, r.name, r.type, r.subtype, r.question_category, r.filetype,
               rc.embedding <=> %s::vector AS dist
        FROM resource_chunks rc
        JOIN resources r ON rc.resource_id = r.id
        WHERE {where_clause}
        ORDER BY dist ASC
        LIMIT %s
        """,
        [query_emb] + params + [limit_chunks],
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()

    best = {}
    for rid, name, rtype_, subtype, qcat, filetype, dist in rows:
        if rid not in best or dist < best[rid][-1]:
            best[rid] = (rid, name, rtype_, subtype, qcat, filetype, dist)
    return sorted(best.values(), key=lambda r: r[-1])


def keyword_match_resources(course_id: int, query: str, rtype: str = None, rfiletype: str = None, limit: int = 20):
    """Real keyword search (Postgres full-text) run alongside the semantic
    search. Catches unique/rare terms that dense embeddings often miss --
    e.g. a specific term that appears in exactly one document."""
    conditions = ["course_id = %s", "raw_text IS NOT NULL"]
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
        SELECT id, name, type, subtype, question_category, filetype,
               ts_rank(to_tsvector('english', raw_text), plainto_tsquery('english', %s)) AS rank
        FROM resources
        WHERE {where_clause}
          AND to_tsvector('english', raw_text) @@ plainto_tsquery('english', %s)
        ORDER BY rank DESC
        LIMIT %s
        """,
        [query] + params + [query, limit],
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


def _format_chunks_with_pages(chunks) -> str:
    lines = []
    for text, name, subtype, page in chunks:
        tag = f"[Source: {name} ({subtype})" + (f", page {page}" if page else "") + "]"
        lines.append(f"{tag}\n{text}")
    return "\n\n".join(lines)


def get_course_context(course_id: int):
    """Returns (name, weightage, grading_policy, exam_dates, target_grade, depth_preference)."""
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """SELECT name, weightage, grading_policy, exam_dates, target_grade, depth_preference
           FROM courses WHERE id = %s""",
        (course_id,),
    )
    row = cur.fetchone()
    cur.close()
    conn.close()
    return row


def _intent_note(course) -> str:
    target_grade, depth_preference = course[4], course[5] or "balanced"
    target_line = f"The student is targeting a grade of {target_grade}%." if target_grade else "The student has no specific target grade -- aim for strong overall mastery."
    depth_line = DEPTH_INSTRUCTIONS.get(depth_preference, DEPTH_INSTRUCTIONS["balanced"])
    return f"{target_line} {depth_line}"


def answer_question(course_id: int, query: str, history: list[dict]) -> str:
    search_query = rewrite_query_for_retrieval(query, history)
    chunks = retrieve_and_rerank(course_id, search_query)
    course = get_course_context(course_id)
    context_str = _format_chunks_with_pages(chunks) or "(No course material has been uploaded yet, or nothing relevant was found.)"

    system_prompt = f"""You are a study assistant for the course "{course[0]}".
Grading breakdown: {course[1]}
Grading policy: {course[2]}
Exam dates: {course[3]}
{_intent_note(course)}

Answer the student's question using ONLY the course material provided below when it's relevant.
When you use it, mention which source it came from (and page number, if given). If the source is
self-written or self-found (not professor-provided), note that it hasn't been verified against
official course content. If the retrieved material doesn't actually answer the question, say so
honestly instead of guessing.

COURSE MATERIAL:
{context_str}
"""

    convo = "\n".join(f"{m['role']}: {m['content']}" for m in history[-6:])
    full_prompt = f"{system_prompt}\n\nConversation so far:\n{convo}\n\nStudent: {query}\nAssistant:"

    response = _generate(full_prompt)
    return response


def get_resource(resource_id: int):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT name, raw_text FROM resources WHERE id = %s", (resource_id,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    return row


def generate_answers_for_question_source(course_id: int, question_source_id: int) -> str:
    source = get_resource(question_source_id)
    if not source or not source[1]:
        return "This question source has no extracted text to work from yet."

    source_name, source_text = source
    course = get_course_context(course_id)

    chunks = retrieve_and_rerank(course_id, source_text[:2000], k_final=8)
    context_str = _format_chunks_with_pages(chunks) or "(No other course material has been uploaded yet.)"

    prompt = f"""You are a study assistant for the course "{course[0]}".
{_intent_note(course)}

Below is a question source named "{source_name}". Answer every question in it,
using the course material provided as grounding where relevant. Show your
reasoning briefly for non-trivial questions, not just the final answer. If a
question can't be answered confidently from the material provided, say so
rather than guessing.

QUESTION SOURCE:
{source_text}

COURSE MATERIAL (for grounding):
{context_str}
"""

    response = _generate(prompt)
    return response


def generate_initial_answer_draft(course_id: int, question_source_id: int) -> tuple[str, str]:
    """First turn of an answer-generation session. Returns (chat_reply, draft_content)."""
    source = get_resource(question_source_id)
    source_name = source[0] if source else "this question source"
    content = generate_answers_for_question_source(course_id, question_source_id)
    reply = f'Here are the generated answers for "{source_name}". Tell me if you\'d like anything changed, or click Finish when you\'re happy with it.'
    return reply, content


def regenerate_answer_draft(course_id: int, question_source_id: int, draft_content: str, feedback_message: str, history: list[dict]) -> tuple[str, str]:
    """A revision turn: takes feedback and produces both a short chat reply and
    the full revised draft. Returns (chat_reply, updated_draft_content)."""
    source = get_resource(question_source_id)
    source_name, source_text = source if source else ("question source", "")
    course = get_course_context(course_id)
    convo = "\n".join(f"{m['role']}: {m['content']}" for m in history[-8:])

    prompt = f"""You are revising a generated answer sheet for "{source_name}" in the
course "{course[0]}", based on student feedback.
{_intent_note(course)}

ORIGINAL QUESTION SOURCE:
{source_text[:2000]}

CURRENT ANSWER DRAFT:
{draft_content}

Conversation so far:
{convo}

STUDENT FEEDBACK:
{feedback_message}

Revise the draft according to the feedback. Respond in EXACTLY this format,
with the literal markers included:
REPLY: <one short sentence confirming what you changed>
---DRAFT---
<the full revised answer document, replacing the entire previous draft>
"""
    response = _generate(prompt)
    text = response.strip()

    if "---DRAFT---" in text:
        reply_part, _, draft_part = text.partition("---DRAFT---")
        reply = reply_part.replace("REPLY:", "").strip()
        new_draft = draft_part.strip()
    else:
        reply = "I've updated the draft."
        new_draft = text

    return reply, new_draft


def generate_discussion_title(user_message: str, assistant_message: str) -> str:
    prompt = f"""Give a short title (3-6 words, no quotes, no punctuation at the end)
that summarizes what this conversation is about. Output ONLY the title.

Student: {user_message}
Assistant: {assistant_message[:400]}
"""
    try:
        response = _generate(prompt)
        title = response.strip().strip('"').strip()
        return title[:80] if title else "New discussion"
    except Exception:
        return "New discussion"


# ---------- Discussion export (doubts / teaching / plan -> documents) ----------

CATEGORY_LABELS = {
    "doubts": "Doubts Clarified",
    "teaching": "Concepts Taught",
    "plan": "Study Plan",
}


def get_discussion_messages(discussion_id: int):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT role, content FROM messages WHERE discussion_id=%s ORDER BY created_at ASC",
        (discussion_id,),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


def _format_transcript(messages) -> str:
    return "\n".join(f"{role.capitalize()}: {content}" for role, content in messages)


def detect_export_categories(discussion_id: int) -> list[str]:
    transcript = _format_transcript(get_discussion_messages(discussion_id))
    if not transcript.strip():
        return []

    prompt = f"""Below is a conversation between a student and a study assistant.
Identify which of these categories genuinely appear in it:
- doubts: the student asked to clarify a doubt, or get a question/solution explained
- teaching: the assistant taught or explained a concept/topic in some depth
- plan: the student asked for a study plan, prep schedule, or practice plan

Respond with ONLY a JSON array of the category keys that apply, e.g. ["doubts","teaching"].
If none clearly apply, respond with [].

CONVERSATION:
{transcript}
"""
    try:
        response = _generate(prompt)
        text = _strip_json_fence(response)
        cats = json.loads(text)
        return [c for c in cats if c in CATEGORY_LABELS]
    except Exception:
        return []


def _strip_json_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    return text.strip()


def generate_topic_tags(text: str) -> list[str]:
    """Read a resource's text and return short subject-matter topic labels
    (not commentary/quality notes -- just what it's actually about)."""
    if not text or not text.strip():
        return []
    prompt = f"""Read the following course material and list the 3-8 main subject-matter
topics it covers. Short labels only (2-4 words each), e.g. ["naive bayes",
"conditional probability"]. Do not include commentary like "important" or
"questions about" -- just the actual subjects.

Respond with ONLY a JSON array of strings.

MATERIAL:
{text[:6000]}
"""
    try:
        response = _generate(prompt)
        tags = json.loads(_strip_json_fence(response))
        return [t.strip().lower() for t in tags if isinstance(t, str) and t.strip()][:8]
    except Exception:
        return []


def generate_export_document(discussion_id: int, categories: list[str]) -> tuple[str, str]:
    transcript = _format_transcript(get_discussion_messages(discussion_id))
    labels = ", ".join(CATEGORY_LABELS[c] for c in categories)

    prompt = f"""Below is a conversation between a student and a study assistant.
Extract and organize ONLY the content relevant to: {labels}.
Ignore unrelated parts of the conversation.

Write it as clean, well-structured study material in Markdown -- headings,
bullet points, numbered steps where helpful. Write it as reference material
a student would study from later, not as a reply to the student (no "sure,
here's..." framing). If more than one category is included, use a top-level
heading per category.

Start your response with exactly one line: TITLE: <a short descriptive title, 4-8 words>
Then a blank line, then the Markdown document.

CONVERSATION:
{transcript}
"""
    response = _generate(prompt)
    text = response.strip()

    if text.startswith("TITLE:"):
        first_line, _, rest = text.partition("\n")
        title = first_line.replace("TITLE:", "").strip()
        body = rest.strip()
    else:
        title = " & ".join(CATEGORY_LABELS[c] for c in categories)
        body = text

    return title, body


# ---------- Exam Mode ----------

def _get_course_resources_for_prep(course_id: int):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """SELECT id, type, subtype, question_category, name, raw_text, created_at
           FROM resources WHERE course_id=%s AND type != 'answer_set'
           ORDER BY created_at DESC""",
        (course_id,),
    )
    resources = cur.fetchall()
    cur.execute(
        """SELECT resource_id, MIN(page_number), MAX(page_number)
           FROM resource_chunks WHERE page_number IS NOT NULL
           GROUP BY resource_id"""
    )
    page_ranges = {r[0]: (r[1], r[2]) for r in cur.fetchall()}
    cur.close()
    conn.close()
    return resources, page_ranges


def get_resource_listing(course_id: int) -> str:
    """Short listing of what's been uploaded -- used so chat-based functions
    at least know what exists, even when they're not doing deep retrieval."""
    resources, page_ranges = _get_course_resources_for_prep(course_id)
    lines = []
    for rid, rtype, subtype, qcat, name, raw_text, created_at in resources:
        line = f"- {rtype}"
        if subtype:
            line += f"/{subtype}"
        if qcat:
            line += f" ({qcat})"
        line += f': "{name}"'
        if rid in page_ranges:
            line += f", pages {page_ranges[rid][0]}-{page_ranges[rid][1]}"
        lines.append(line)
    return "\n".join(lines) or "(no resources uploaded yet)"


def exam_intake_reply(course_id: int, message: str, history: list[dict]) -> str:
    """Interviewer role: gather constraints before a prep sheet can be generated."""
    course = get_course_context(course_id)
    resource_listing = get_resource_listing(course_id)

    system_prompt = f"""You are helping a student prepare for an exam in "{course[0]}".
Grading breakdown: {course[1]}
Exam dates on record: {course[3]}
{_intent_note(course)}

Course material already uploaded (you can reference these by name if relevant
to the conversation):
{resource_listing}

Your job right now is ONLY to understand their situation well enough to build a
prep checklist -- do not start teaching or answering content questions yet.
Ask about: how much time they have left, which exam/topics this covers, how many
hours per day they can realistically study, and what they already feel weak or
confident on. Ask naturally, a question or two at a time, not a giant list.
Once you feel you have enough to work with, tell them they can click
"Generate prep sheet" whenever they're ready.
"""
    convo = "\n".join(f"{m['role']}: {m['content']}" for m in history[-10:])
    full_prompt = f"{system_prompt}\n\nConversation so far:\n{convo}\n\nStudent: {message}\nAssistant:"

    response = _generate(full_prompt)
    return response


def exam_advisor_reply(course_id: int, message: str, history: list[dict], tasks: list[dict]) -> str:
    """Once a prep sheet exists: answer questions (grounded in real material),
    or explain what a change would look like -- but never silently mutate the
    list from chat."""
    course = get_course_context(course_id)
    resource_listing = get_resource_listing(course_id)
    search_query = rewrite_query_for_retrieval(message, history)
    chunks = retrieve_and_rerank(course_id, search_query)
    grounded_excerpts = _format_chunks_with_pages(chunks) or "(nothing closely relevant retrieved)"
    task_list_str = "\n".join(f"{i+1}. [{'x' if t['done'] else ' '}] {t['instruction']}" for i, t in enumerate(tasks)) or "(no tasks yet)"

    system_prompt = f"""You are helping a student with their exam prep checklist for "{course[0]}".
{_intent_note(course)}

Course material already uploaded:
{resource_listing}

Current checklist:
{task_list_str}

Relevant excerpts from the student's course material, if their message needs it:
{grounded_excerpts}

Answer the student's question, give guidance on how to do a task, explain a
concept using the material above, or justify why something is ordered/included
the way it is. If they ask you to change, add, or remove something, explain
clearly what you'd change -- but do NOT claim you've updated the list. Tell
them to click "Update prep sheet" to actually apply it once they're ready.
"""
    convo = "\n".join(f"{m['role']}: {m['content']}" for m in history[-10:])
    full_prompt = f"{system_prompt}\n\nConversation so far:\n{convo}\n\nStudent: {message}\nAssistant:"

    response = _generate(full_prompt)
    return response



def generate_prep_sheet(course_id: int, discussion_id: int) -> list[dict]:
    course = get_course_context(course_id)
    transcript = _format_transcript(get_discussion_messages(discussion_id))
    resources, page_ranges = _get_course_resources_for_prep(course_id)

    resource_lines = []
    question_source_blocks = []
    for rid, rtype, subtype, qcat, name, raw_text, created_at in resources:
        line = f'- id={rid}, type={rtype}'
        if subtype:
            line += f"/{subtype}"
        if qcat:
            line += f" ({qcat})"
        line += f', name="{name}"'
        if rid in page_ranges:
            line += f", pages {page_ranges[rid][0]}-{page_ranges[rid][1]}"
        line += f", added {created_at.strftime('%b %d')}"
        resource_lines.append(line)
        if rtype == "question_source" and raw_text:
            question_source_blocks.append(f'--- "{name}" ---\n{raw_text[:3000]}')

    resource_listing = "\n".join(resource_lines) or "(no resources uploaded yet)"
    question_sources_block = "\n\n".join(question_source_blocks) or "(none)"

    chunks = retrieve_and_rerank(course_id, transcript[-2000:], k_final=12)
    grounded_excerpts = _format_chunks_with_pages(chunks) or "(no material retrieved)"

    prompt = f"""You are creating an exam preparation checklist for a student, based on a
conversation where you gathered their time constraints and needs.

Course: {course[0]}
Grading weightage: {course[1]}
{_intent_note(course)}

INTAKE CONVERSATION:
{transcript}

AVAILABLE COURSE MATERIAL (reference these by their EXACT name -- never invent
a resource that isn't listed here):
{resource_listing}

QUESTION SOURCE CONTENTS (for referencing specific questions/tutorials accurately):
{question_sources_block}

GROUNDED EXCERPTS WITH PAGE NUMBERS (only cite a page number if it appears
here -- never invent one; if no page number is available for a resource,
reference it by name/topic instead):
{grounded_excerpts}

Create an ORDERED checklist of concrete, actionable tasks that fits the
student's time constraints, prioritized by the course's grading weightage
and what they said they're weak on. Examples of good task phrasing:
"Read pages 33-35 in Textbook.pdf and memorize the key definitions",
"Attempt questions 4-6 from Tutorial 2", "Review your notes on recursion
(exported Mar 20)". Order for retention -- foundational material before
material that builds on it, and don't cram everything tested last into the end.

Respond with ONLY a JSON array, no other text, in this exact shape:
[
  {{"instruction": "...", "resource_name": "<exact name from the list above, or null>", "question_source_name": "<exact name of a question_source if this task is practicing questions from it, or null>"}}
]
"""
    response = _generate(prompt)
    text = _strip_json_fence(response)

    try:
        tasks_raw = json.loads(text)
    except Exception:
        tasks_raw = []

    name_to_id = {name: rid for rid, rtype, subtype, qcat, name, raw_text, created_at in resources}
    qsource_name_to_id = {
        name: rid for rid, rtype, subtype, qcat, name, raw_text, created_at in resources if rtype == "question_source"
    }

    tasks = []
    for t in tasks_raw:
        instruction = (t.get("instruction") or "").strip()
        if not instruction:
            continue
        tasks.append({
            "instruction": instruction,
            "resource_id": name_to_id.get(t.get("resource_name")),
            "question_source_id": qsource_name_to_id.get(t.get("question_source_name")),
        })
    return tasks


def regenerate_prep_sheet(course_id: int, discussion_id: int, current_tasks: list[dict]) -> list[dict]:
    """Re-reads the conversation + current tasks and produces an updated task
    list. Tasks the model says are unchanged should keep their existing id
    (passed back) so the caller can preserve their `done` status; genuinely
    new tasks get id=null."""
    course = get_course_context(course_id)
    transcript = _format_transcript(get_discussion_messages(discussion_id))
    resources, page_ranges = _get_course_resources_for_prep(course_id)

    resource_lines = []
    for rid, rtype, subtype, qcat, name, raw_text, created_at in resources:
        line = f'- id={rid}, type={rtype}, name="{name}"'
        if rid in page_ranges:
            line += f", pages {page_ranges[rid][0]}-{page_ranges[rid][1]}"
        resource_lines.append(line)
    resource_listing = "\n".join(resource_lines) or "(no resources uploaded yet)"

    current_list_str = "\n".join(
        f'id={t["id"]}: {t["instruction"]}{" [DONE]" if t["done"] else ""}' for t in current_tasks
    ) or "(no tasks yet)"

    prompt = f"""Below is an exam prep checklist and the full conversation the student has
had about it (including any requests to change it).

Course: {course[0]}
{_intent_note(course)}

CURRENT CHECKLIST:
{current_list_str}

FULL CONVERSATION:
{transcript}

AVAILABLE COURSE MATERIAL:
{resource_listing}

Produce the UPDATED checklist reflecting anything the student asked to change,
add, or remove in the conversation. For tasks that should stay exactly as they
are, keep their original "id". For tasks that are genuinely new, use "id": null.
Do NOT include an id for a task that no longer applies (i.e. just omit it).
Tasks marked [DONE] should generally stay unless the student explicitly asked
to remove/change that exact one.

Respond with ONLY a JSON array, no other text:
[
  {{"id": <existing id or null>, "instruction": "...", "resource_name": "<exact name or null>", "question_source_name": "<exact name or null>"}}
]
"""
    response = _generate(prompt)
    text = _strip_json_fence(response)

    try:
        tasks_raw = json.loads(text)
    except Exception:
        return current_tasks  # generation failed -- leave the list untouched

    name_to_id = {name: rid for rid, rtype, subtype, qcat, name, raw_text, created_at in resources}
    qsource_name_to_id = {
        name: rid for rid, rtype, subtype, qcat, name, raw_text, created_at in resources if rtype == "question_source"
    }

    updated = []
    for t in tasks_raw:
        instruction = (t.get("instruction") or "").strip()
        if not instruction:
            continue
        updated.append({
            "id": t.get("id"),
            "instruction": instruction,
            "resource_id": name_to_id.get(t.get("resource_name")),
            "question_source_id": qsource_name_to_id.get(t.get("question_source_name")),
        })
    return updated
