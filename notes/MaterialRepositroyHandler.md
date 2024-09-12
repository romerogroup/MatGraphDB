# MaterialRepositoryHandler Notes


### `create_parquet_from_data` Implementation Notes

- We opted for users returning a tuple (column_names, values) instead of dictionaries for performance reasons.
- Returning dictionaries resulted in a 30x performance degradation due to the need for dynamic key/value appending in nested loops.
- Considered but rejected approaches:
  - Returning just the values (without column names), but it would have required the user to define a schema or strictly follow an order.
  - Returning dictionaries would simplify logic but had a significant performance impact.
