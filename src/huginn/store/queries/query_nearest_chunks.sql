select
  c.chunk_id,
  sf.path as source_path,
  c.page_start,
  c.page_end,
  c.text,
  vc.distance
from vec_chunks vc
join chunks c on c.chunk_id = vc.chunk_id
join documents d on d.document_id = c.document_id
join source_files sf on sf.source_file_id = d.source_file_id
where vc.embedding match ? and k = ?
order by vc.distance asc, c.chunk_id asc
