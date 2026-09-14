-- Run this in the Supabase SQL Editor on your EXISTING project.
-- Adds Intent settings, Exam Mode, and per-page tracking for PDFs.

-- Intent settings (per course)
alter table courses add column if not exists target_grade numeric;
alter table courses add column if not exists depth_preference text default 'balanced';  -- efficient | balanced | deep

-- Exam Mode discussions are regular discussions with a mode flag
alter table discussions add column if not exists mode text default 'general';  -- general | exam_prep

-- Per-page tracking so PDF page citations are real, not guessed
alter table resource_chunks add column if not exists page_number integer;

-- The prep-sheet todo list
create table if not exists exam_prep_tasks (
  id serial primary key,
  discussion_id integer references discussions(id) on delete cascade,
  order_index integer not null,
  instruction text not null,
  linked_resource_id integer references resources(id) on delete set null,
  question_source_id integer references resources(id) on delete set null,
  done boolean default false,
  created_at timestamptz default now()
);
