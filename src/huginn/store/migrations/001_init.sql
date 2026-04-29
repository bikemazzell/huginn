create table if not exists source_files (
  source_file_id integer primary key,
  path text not null unique,
  sha256 text not null,
  file_type text not null,
  modified_at text not null,
  status text not null,
  error_message text
);

create table if not exists documents (
  document_id integer primary key,
  source_file_id integer not null references source_files(source_file_id) on delete cascade,
  title text not null,
  page_count integer not null,
  extracted_text_hash text not null
);

create table if not exists pages (
  page_id integer primary key,
  document_id integer not null references documents(document_id) on delete cascade,
  page_number integer not null,
  text text not null
);

create table if not exists chunks (
  chunk_id integer primary key,
  document_id integer not null references documents(document_id) on delete cascade,
  page_start integer not null,
  page_end integer not null,
  chunk_index integer not null,
  text text not null,
  token_count integer not null
);

create table if not exists chunk_embeddings (
  chunk_id integer primary key references chunks(chunk_id) on delete cascade,
  vector_json text not null
);
