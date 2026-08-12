# AGENTS.md

Quick-start rules for coding agents working in this repository. Follow
[`CODING-STYLE.md`](./CODING-STYLE.md) for detailed engineering and test guidance.

## Project boundary

`seqcache` is a backend-independent Rust manager for paged inference state. It owns logical page
lifetimes, retained-prefix indexing, strict admission, append transactions, and accounting. Runtime
implementations own physical CPU or accelerator storage, native page tables, synchronization, and
reclamation through `PageBackend`.

Keep the crate useful for real KV caches and other position-indexed inference state without adding
abstractions for hypothetical consumers. Prefer one coherent current API over adapters,
compatibility wrappers, deprecated aliases, or parallel legacy paths.

## Permission boundaries

Do not perform state-changing Git operations without explicit human permission. This includes
staging, commits, amends, branch changes, merges, rebases, cherry-picks, resets, stashes, tags, and
pushes. Read-only Git inspection is allowed.

Do not publish crates, tags, releases, benchmark results, or repository changes without an explicit
request. Ordinary local formatting, builds, tests, Clippy, package dry-runs, CPU examples, and
micromeasures are allowed when relevant. Ask before running GPU work if another model or
memory-heavy process may be active.

## Engineering policy

- Read the relevant implementation, public contract, and conformance tests before editing.
- Preserve unrelated work in a dirty tree.
- Keep logical ownership in `SequenceCache` and physical storage and synchronization in
  `PageBackend`.
- Preserve atomic append reservation, commit, abort, page-table publication, and retirement
  semantics on every failure path.
- Keep admission and reserved-page accounting exact. Deferred pages remain charged until reclaimed.
- Do not recycle storage which outstanding CPU or accelerator work can still access.
- Keep prefix keys backend-agnostic. The prefix index intentionally uses
  [`rart`](https://github.com/rdaum/rart-rs) for compact longest-prefix lookup over interned token
  blocks.
- Add or update focused tests for changed behaviour and regression cases.
- Measure performance claims with the micromeasure suite and, when applicable, a production-shaped
  inference workload.
- Update README, examples, rustdoc, and the `[Unreleased]` changelog when public behaviour changes.
- Use Canadian English in documentation and comments.

## Repository shape

```text
src/          # public facade, backend contract, manager, prefix index, errors, and metrics
tests/        # failure-injecting conformance and state-machine coverage
benches/      # micromeasure manager and prefix-index lifecycles
examples/     # complete CPU backend and feature-gated CUDA backend
```

## Verification

For ordinary Rust changes, run:

```bash
dprint check
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
cargo package
```

Run `cargo run --example cpu_paged_kv` after backend-contract changes. Run
`cargo run --features cuda-example --example cuda_paged_state` on a CUDA-capable Linux host for
accelerator-facing changes. State environmental gaps explicitly.

## Commits

Only create commits when explicitly requested. Use Conventional Commits with a specific, imperative
subject. Non-trivial commits should explain motivation, material behaviour, and validation or known
limitations without restating the file list.
