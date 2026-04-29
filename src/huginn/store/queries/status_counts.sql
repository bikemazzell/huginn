select
  (select count(*) from source_files) as source_file_count,
  (select count(*) from documents) as document_count,
  (select count(*) from chunks) as chunk_count
