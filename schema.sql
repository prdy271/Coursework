-- Run this entire file once in the Supabase SQL Editor for a BRAND NEW project.
-- If you already have a project from an earlier version, use the migration_vN.sql files instead.

create extension if not exists vector;
create extension if not exists pg_trgm;

create table if not exists courses (
  id serial primary key,
  name text not null,
  code text,
  instructor text,
  venue text,
  weightage jsonb default '{}',
  grading_policy text,
  exam_dates jsonb default '[]',
  target_grade numeric,
  depth_preference text default 'balanced',  -- efficient | balanced | deep
  created_at timestamptz default now()
);

create table if not exists resources (
  id serial primary key,
  course_id integer references courses(id) on delete cascade,
  type text not null,               -- book | note | question_source | answer_set
  subtype text,                     -- prof_provided | self_found | self_written | model_generated | from_discussion
  question_category text,           -- tutorial | assignment | mini_test | pyq | pyexam | quiz | custom_generated
  linked_question_source_id integer references resources(id) on delete set null,
  name text,
  filetype text,
  file_size integer,
  file_path text,
  raw_text text,
  topics text[],
  created_at timestamptz default now()
);

create table if not exists resource_chunks (
  id serial primary key,
  resource_id integer references resources(id) on delete cascade,
  chunk_text text,
  chunk_index integer,
  page_number integer,
  embedding vector(384)
);

create table if not exists discussions (
  id serial primary key,
  course_id integer references courses(id) on delete cascade,
  title text not null default 'Untitled discussion',
  mode text default 'general',      -- general | exam_prep | answer_gen
  target_question_source_id integer references resources(id) on delete set null,
  draft_content text,
  created_at timestamptz default now(),
  updated_at timestamptz default now()
);

create table if not exists messages (
  id serial primary key,
  discussion_id integer references discussions(id) on delete cascade,
  role text not null,
  content text not null,
  feedback smallint,
  created_at timestamptz default now()
);

create table if not exists events (
  id serial primary key,
  course_id integer references courses(id) on delete cascade,
  name text not null,
  category text not null,
  ref_type text,
  ref_id integer,
  created_at timestamptz default now()
);

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

create index if not exists resource_chunks_embedding_idx
  on resource_chunks using ivfflat (embedding vector_cosine_ops) with (lists = 100);

create index if not exists resources_name_trgm_idx on resources using gin (name gin_trgm_ops);

create index if not exists resources_raw_text_fts_idx
  on resources using gin (to_tsvector('english', coalesce(raw_text, '')));

create index if not exists resource_chunks_text_fts_idx
  on resource_chunks using gin (to_tsvector('english', chunk_text));
