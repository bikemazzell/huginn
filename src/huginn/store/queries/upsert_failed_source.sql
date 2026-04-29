insert into source_files(path, sha256, file_type, modified_at, status, error_message)
values (?, ?, ?, ?, ?, ?)
on conflict(path) do update set
  sha256 = excluded.sha256,
  file_type = excluded.file_type,
  modified_at = excluded.modified_at,
  status = excluded.status,
  error_message = excluded.error_message
