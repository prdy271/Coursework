-- Run this in the Supabase SQL Editor on your EXISTING project.
-- Adds file_path so uploaded resources can be downloaded/deleted from the UI.

alter table resources add column if not exists file_path text;
