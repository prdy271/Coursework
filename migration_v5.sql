-- Run this in the Supabase SQL Editor on your EXISTING project.
-- Supports the new iterative "Generate Answers" flow (chat + live draft + finish).

alter table discussions add column if not exists target_question_source_id integer references resources(id) on delete set null;
alter table discussions add column if not exists draft_content text;
