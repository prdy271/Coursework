-- Run this in the Supabase SQL Editor on your EXISTING project.
-- Supports: message feedback (B5), topic tagging (C6), and chunk-level
-- keyword search used by the re-ranking pass (A2).

alter table messages add column if not exists feedback smallint;  -- 1 = up, -1 = down, null = none
alter table resources add column if not exists topics text[];      -- topic tags assigned at creation time

create index if not exists resource_chunks_text_fts_idx
  on resource_chunks using gin (to_tsvector('english', chunk_text));
