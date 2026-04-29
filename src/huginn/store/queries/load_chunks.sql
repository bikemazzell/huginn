select
  c.chunk_id,
  sf.path as source_path,
  c.page_start,
  c.page_end,
  c.text,
  ce.vector_json
from chunks c
join documents d on d.document_id = c.document_id
join source_files sf on sf.source_file_id = d.source_file_id
join chunk_embeddings ce on ce.chunk_id = c.chunk_id
order by c.chunk_id asc
