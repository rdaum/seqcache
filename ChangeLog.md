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
- Document the public backend contract, lifecycle invariants, project style, and contribution
  checks.
