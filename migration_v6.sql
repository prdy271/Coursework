-- Run this in the Supabase SQL Editor on your EXISTING project.
-- Enables fuzzy (typo-tolerant) filename search for the Resources search bar.

create extension if not exists pg_trgm;
create index if not exists resources_name_trgm_idx on resources using gin (name gin_trgm_ops);
