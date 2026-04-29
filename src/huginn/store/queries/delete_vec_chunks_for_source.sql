delete from vec_chunks
where chunk_id in (
  select c.chunk_id
  from chunks c
  join documents d on d.document_id = c.document_id
  where d.source_file_id = ?
)
