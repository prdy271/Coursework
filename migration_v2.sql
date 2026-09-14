-- Run this in the Supabase SQL Editor on your EXISTING project.
-- It adds new structure without touching your existing courses/resources/data.

-- 1. Discussions: a course can now have many separate discussion threads,
--    instead of one single long chat.
create table if not exists discussions (
  id serial primary key,
  course_id integer references courses(id) on delete cascade,
  title text not null default 'Untitled discussion',
  created_at timestamptz default now(),
  updated_at timestamptz default now()
);

-- 2. Move messages onto discussions instead of directly onto courses.
alter table messages add column if not exists discussion_id integer references discussions(id) on delete cascade;

-- Backfill: wrap any existing messages (from the old single-chat design) into
-- one "General discussion" per course so nothing is lost.
do $$
declare
  c record;
  new_discussion_id integer;
begin
  for c in select distinct course_id from messages where discussion_id is null loop
    insert into discussions (course_id, title) values (c.course_id, 'General discussion')
    returning id into new_discussion_id;

    update messages set discussion_id = new_discussion_id
    where course_id = c.course_id and discussion_id is null;
  end loop;
end $$;

-- 3. Question-source subcategories (tutorial, assignment, PYQ, etc.) and a
--    link from generated answer sets back to the question source they answer.
alter table resources add column if not exists question_category text;
alter table resources add column if not exists linked_question_source_id integer references resources(id) on delete set null;

-- 4. Timeline events, colour-coded by category.
create table if not exists events (
  id serial primary key,
  course_id integer references courses(id) on delete cascade,
  name text not null,
  category text not null,  -- data_upload | discussion | answer_generation
  ref_type text,           -- 'resource' | 'discussion' (what this event points to)
  ref_id integer,
  created_at timestamptz default now()
);
