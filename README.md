# Nexir

A Rust project for timeline management.

## Project Structure

- `src/lib.rs` - Library root
- `src/timeline/` - Timeline module
  - `ids.rs` - ID types
  - `store.rs` - Storage implementation
  - `effect.rs` - Effects
  - `track.rs` - Track management
  - `source.rs` - Source management
  - `rational.rs` - Rational number utilities
  - `query.rs` - Query functionality
  - `mutation.rs` - Mutation operations
  - `transform.rs` - Transformations
  - `tests.rs` - Tests

## Building

```bash
cargo build
```

## Running Tests

```bash
cargo test
```
