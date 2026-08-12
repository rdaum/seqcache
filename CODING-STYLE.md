# Coding Guidelines

## Engineering priorities

`seqcache` owns logical page lifetimes, prefix sharing, admission reservations, and transactional
mutation for inference runtimes. Bugs at this boundary can expose invalid model state, recycle pages
which are still in use, or admit work for which physical capacity does not exist.

When priorities conflict, use this order:

1. Correct ownership, atomic state transitions, and exact accounting.
2. Clear backend contracts and actionable failures.
3. Measured performance on representative inference lifecycles.
4. API and implementation convenience.

Keep the crate backend-independent without designing abstractions for hypothetical runtimes. A new
generic boundary should remove demonstrated duplication or express an invariant which current
backends need.

## Rust style

- The crate uses Rust 2024 and the minimum toolchain declared by `rust-version` in `Cargo.toml`.
- Format Rust with default `rustfmt` and prose, TOML, and JSON with `dprint`:

  ```bash
  cargo fmt --all
  dprint fmt
  ```

- Keep imports at the top of each module. Group standard-library and external-crate imports where it
  improves readability.
- Modules and functions use `snake_case`; types and traits use `PascalCase`; constants use
  `SCREAMING_SNAKE_CASE`.
- Names describe current behaviour. Avoid historical names such as `new`, `old`, `legacy`, `v2`, or
  `temporary` unless the distinction is an actual public contract.
- Prefer concrete types and direct functions until an abstraction clarifies ownership, removes real
  duplication, or establishes a useful test boundary.
- Keep public types and capacity units explicit. Avoid boolean arguments whose meaning is unclear at
  the call site.
- Prefer early returns, `let else`, guarded matches, and focused helpers over deeply nested control
  flow. Do not split code into tiny functions merely to reduce line count.

Use Canadian English in documentation and comments: `behaviour`, `centralise`, and `-re` endings,
while retaining conventional technical spellings and API names.

## Ownership and transactions

The manager and backend divide responsibility deliberately:

- `SequenceCache` owns logical pages, sequence positions, prefix references, reservations, and
  managed-byte accounting.
- `PageBackend` owns physical storage, backend-native page tables, synchronization, rollback state,
  and reclamation.

Preserve these invariants:

- Only a private final page may be partially valid and writable.
- Sealed pages are immutable while shared or retained.
- Admission accounts for every page required up to the request's declared maximum position.
- An append reservation publishes every destination page before model execution.
- Commit and abort apply to the whole reservation. Backend failure must not expose a partially
  updated logical table or partially recycled page set.
- A stale sequence, page, prefix, or reservation handle must never become valid through slot reuse.
- Prefix eviction must not invalidate pages referenced by live sequences.
- Deferred retirement remains charged until the backend reports the pages reusable.

Keep failure handling explicit. A backend method which promises atomicity must either complete the
operation or preserve enough state for the caller to retry or abort. Panics are reserved for proven
internal invariants; recoverable capacity, validation, allocation, publication, synchronization, and
retirement failures return precise errors.

## Accelerator and unsafe code

- Keep CUDA stream ordering explicit. Work on a non-blocking stream does not synchronize with the
  default stream.
- Do not shorten a published table, restore overwritten contents, or recycle physical storage until
  every operation which can access it has completed.
- Prefer deferred event-based reclamation in production backends when host synchronization would be
  material, while keeping the accounting contract exact.
- Write model output directly into reserved physical pages. Do not introduce a large contiguous
  staging cache followed by a scatter copy merely to simplify the backend.
- Keep unsafe code narrow and behind a safe ownership boundary. Document the pointer, lifetime,
  allocation, and stream-ordering assumptions which make non-obvious unsafe operations valid.
- Validate sizes, offsets, conversions, and driver results before exposing them to safe code.

## Public API

- Maintain one coherent current API. Do not add adapters, deprecated aliases, compatibility bridges,
  or parallel legacy paths unless an existing supported consumer has a concrete requirement.
- Public names describe semantics rather than the implementation which first required them.
- Preserve backend and key agnosticism where it is real. The manager may support transformer KV,
  compressed attention, or recurrent snapshots without pretending their physical layouts are the
  same.
- Document ownership transfer, atomicity, retry behaviour, synchronization responsibility, and byte
  accounting for public operations.
- Treat a breaking change as an intentional design decision. Update examples, conformance tests,
  rustdoc, README guidance, and the changelog together.

## Performance work

Performance claims require measurements. Relevant quantities include reservation and commit cost,
prefix lookup and restoration cost, allocation and reclamation overhead, page-table publication, and
end-to-end inference throughput.

- Establish a representative baseline before optimizing.
- Use the micromeasure suite for manager and index overhead, and a production-shaped runtime for
  backend or end-to-end claims.
- Preserve correctness validation around benchmarks. Faster code which changes reservation geometry,
  cache hits, committed rows, or reclamation behaviour is not an improvement.
- Report workload, build profile, hardware, sample count, and relevant cache state with results.
- Keep hot-path allocation, synchronization, copying, and indirection visible and intentional.
- Do not infer an inference-runtime win solely from a manager or kernel micromeasure.

## Dependencies

Every dependency expands the build and maintenance surface.

- Prefer the standard library and existing dependencies where they are a reasonable fit.
- Before adding a crate, inspect default features, the transitive graph, maintenance state, licence,
  platform support, and minimum Rust version.
- Avoid dependencies for small helpers or wrappers which are clearer to implement directly.
- Keep `rart` as the prefix index because longest-prefix search and compact shared paths match the
  retained-prefix model. Do not replace it without conformance coverage and representative lookup,
  insertion, and eviction measurements.

## Testing

Tests should make a specific incorrect implementation fail.

- Add focused regression tests for bugs and behavioural tests for new contracts.
- Exercise allocation, publication, commit, abort, retirement, and restoration failures—not only the
  successful lifecycle.
- Recompute invariants with `SequenceCache::validate` after meaningful state transitions in tests.
- Cover exact page boundaries, partial tails, multi-page spans, shared prefixes, copy-on-write,
  generational staleness, and accounting changes.
- Keep tests deterministic, host-independent, and free of debug output. Environment-backed CUDA
  validation supplements rather than replaces the CPU conformance suite.

The standard verification baseline is:

```bash
dprint check
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features --lib --tests
cargo package
```

Run the CPU example when changing the backend contract. Run the CUDA example on a CUDA-capable Linux
host when changing accelerator-facing behaviour:

```bash
cargo run --example cpu_paged_kv
cargo run --features cuda-example --example cuda_paged_state
```

State exactly what was and was not run when environmental requirements prevent full validation.

## Documentation and change history

- Documentation and comments describe current behaviour. Put useful history in `ChangeLog.md` or a
  focused design note.
- Begin each hand-written Rust source unit with module-level rustdoc explaining what the module
  owns.
- Keep `lib.rs` as a readable public facade and split implementation by cohesive ownership or policy
  boundaries.
- Keep examples executable and update them whenever the backend contract changes.
- Add notable changes to the `[Unreleased]` section of `ChangeLog.md`. Write for a downstream
  runtime author deciding whether and how to upgrade, not as a raw commit log.
- Distinguish implemented behaviour, measured results, targets, and proposals.

## Agent-assisted contributions

AI tools are acceptable, but the contributor remains responsible for every change.

- Review and understand generated code before committing it.
- Do not submit code, tests, benchmarks, or claims which cannot be explained and verified from
  repository evidence.
- Remove filler, marketing language, and comments which merely restate the code.
- Keep commits cohesive and create them only when the human explicitly requests one.
- Use Conventional Commits with a specific, imperative subject. Give non-trivial commits a body
  which explains motivation, material behaviour, and validation or remaining limitations.

## Review checklist

1. Are logical and physical ownership responsibilities unambiguous?
2. Are commit, abort, failure, and retry transitions atomic and recoverable?
3. Is admission and managed-byte accounting exact across the complete lifecycle?
4. Can any outstanding CPU or accelerator operation observe recycled storage?
5. Does every new test guard a meaningful contract or regression?
6. Are performance claims backed by the appropriate micromeasure and runtime evidence?
7. Does the change avoid compatibility scaffolding, unnecessary abstraction, allocation, and
   dependencies?
8. Did dprint, rustfmt, strict Clippy, tests, examples, and package validation pass where
   applicable?
9. Are rustdoc, README guidance, examples, and `ChangeLog.md` consistent with the code?
