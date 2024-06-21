find data/processed/object_found/ -type f -name "*.jpg" -print0 | xargs -0 rm
find data/processed/no_object_found/ -type f -name "*.jpg" -print0 | xargs -0 rm
find data/processed/log/ -type f -name "*.json" -print0 | xargs -0 rm
find data/to_sort/ -type f -name "*.jpg" -print0 | xargs -0 rm
