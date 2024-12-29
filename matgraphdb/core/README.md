# Design Reasons

- Seperate Node and Edge operations
- NodeStore always deals with node features, does not need source_id or target_id
- EdgeStore always deals with edge features, does need source_id or target_id, plus any edge-specific attributes (weights, timestamps, labels, etc.).s
