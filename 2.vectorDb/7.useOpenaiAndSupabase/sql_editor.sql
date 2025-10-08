-- 1) 擴充功能：uuid 與向量型別
create extension if not exists pgcrypto;
create extension if not exists vector;

-- 2) 向量表：存文字、metadata 與向量
create table if not exists public.documents (
  id uuid primary key default gen_random_uuid(),
  content text not null,
  metadata jsonb,
  embedding vector(1536) not null
);

-- 3) 近鄰搜尋索引
create index if not exists documents_embedding_ivfflat
on public.documents using ivfflat (embedding vector_cosine_ops)
with (lists = 100);

-- 4) 檢索用 RPC
create or replace function public.match_documents_v2 (
  query_embedding vector(1536),
  match_count int default 10,
  filter jsonb default '{}'::jsonb
)
returns table (
  id uuid,
  content text,
  metadata jsonb,
  similarity float
)
language sql stable
as $$
  select
    d.id,
    d.content,
    d.metadata,
    -- pgvector 的 <=> 是「距離」；用 cosine 時，1 - 距離 = 相似度
    1 - (d.embedding <=> query_embedding) as similarity
  from public.documents d
  -- 可選的 metadata 過濾；filter 例如 {"source":"lecture"}
  where (filter = '{}'::jsonb) or (d.metadata @> filter)
  order by d.embedding <=> query_embedding
  limit match_count;
$$;

-- 5) 權限策略
alter table public.documents disable row level security;

-- 6) 刷新 PostgREST 快取
select pg_notify('pgrst','reload schema');
select pg_notify('pgrst','reload config');

-- 7) 更新統計
analyze public.documents;