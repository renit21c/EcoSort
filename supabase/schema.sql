-- EcoSort AI: run in Supabase SQL Editor.
create table if not exists public.classifications (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  label text not null check (label in ('battery','biological','brown-glass','cardboard','clothes','green-glass','metal','paper','plastic','shoes','trash','white-glass')),
  confidence numeric not null check (confidence >= 0 and confidence <= 1),
  probabilities jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);
create index if not exists classifications_user_created_idx on public.classifications(user_id, created_at desc);
alter table public.classifications enable row level security;
drop policy if exists "Users can read own classifications" on public.classifications;
create policy "Users can read own classifications" on public.classifications for select using (auth.uid() = user_id);
drop policy if exists "Users can insert own classifications" on public.classifications;
create policy "Users can insert own classifications" on public.classifications for insert with check (auth.uid() = user_id);
drop policy if exists "Users can delete own classifications" on public.classifications;
create policy "Users can delete own classifications" on public.classifications for delete using (auth.uid() = user_id);
