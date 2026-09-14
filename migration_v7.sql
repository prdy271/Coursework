-- Run this in the Supabase SQL Editor on your EXISTING project.
-- Improves semantic search: correct distance metric + real keyword search
-- fused with it (hybrid search), so unique/rare terms are reliably found.

-- Cosine distance is the correct metric for these embeddings, not L2.
drop index if exists resource_chunks_embedding_idx;
create index if not exists resource_chunks_embedding_idx
  on resource_chunks using ivfflat (embedding vector_cosine_ops) with (lists = 100);

-- Full-text index so literal/unique keywords are found even when the
-- embedding model's semantic signal for them is weak.
create index if not exists resources_raw_text_fts_idx
  on resources using gin (to_tsvector('english', coalesce(raw_text, '')));
