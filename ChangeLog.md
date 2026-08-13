# Changelog

All notable changes to this project will be documented in this file.

The changelog follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and releases follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Establish `seqcache` as a standalone backend-independent crate for paged inference-state
  ownership, retained prefixes, strict admission, transactional multi-page appends, and exact
  accounting.
- Add failure-injecting conformance tests, deterministic state-machine validation, micromeasure
  benchmarks, and complete CPU and CUDA backend examples.
- Expand micromeasure coverage with focused manager reads, reservation access, admission, sequence
  finish, prefix-index mutation and lookup scaling, accounting refresh, and invariant validation.
- Add a feature-gated native `wgpu` backend example with a storage-buffer page slab, a
  device-resident logical page table, and submission-safe page recycling.
- Add canonical ordered batch append reservation, access, commit, and abort operations. Partial
  backend failures return the exact reservations which remain pending for retry or rollback.
- Document the public backend contract, lifecycle invariants, project style, and contribution
  checks.

### Changed

- Remove cache-size-dependent work from routine accounting, maintain an ordered LRU prefix index,
  use faster randomized token-block hashing, and avoid quadratic page-membership checks during
  sequence finish and prefix eviction.
