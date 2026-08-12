//! Focused manager and prefix-index micromeasures over a storage-free backend.
//!
//! Primitive cases isolate manager work where the public API permits it;
//! lifecycle cases keep state stable or report the populated fixture size so
//! whole-cache accounting scans remain visible in the result.

use micromeasure::{
    BenchContext, BenchSampleResult, BenchmarkMainOptions, BenchmarkRuntimeOptions,
    ComparisonPolicy, MetricValue, Throughput, black_box, run_benchmark_main,
};
use seqcache::{
    AdmissionOutcome, AdmissionRequest, AppendReservation, BackendAppendCommit, BackendAppendPage,
    CacheConfig, PageAllocation, PageBackend, PageId, PrefixEntryId, PrefixMatch, RetainOutcome,
    RetireError, RetireOutcome, SequenceCache, SequenceId,
};
use std::convert::Infallible;
use std::time::Duration;

const PAGE_TOKENS: usize = 128;
const PAGE_BYTES: usize = 4096;
const MANAGED_BYTES: usize = 256 << 20;
const OBSERVED_PREFIX_TOKENS: usize = 4_736;
const OBSERVED_PROMPT_TOKENS: usize = 5_433;
const FIXED_PREFIX_MUTATION_OPERATIONS: usize = 128;
const FIXED_FINISH_OPERATIONS: usize = 1_024;

#[derive(Clone, Copy)]
struct BenchPage(u64);

#[derive(Default)]
struct BenchBackend {
    next_page: u64,
    recycled: Vec<BenchPage>,
}

#[derive(Default)]
struct BenchBackendContext {
    table: Vec<u64>,
    position: usize,
}

impl PageBackend for BenchBackend {
    type Page = BenchPage;
    type Context<'a> = BenchBackendContext;
    type AppendTransaction = (usize, usize);
    type Error = Infallible;

    fn page_bytes(&self) -> usize {
        PAGE_BYTES
    }

    fn allocate_page(
        &mut self,
        _context: &mut Self::Context<'_>,
    ) -> Result<PageAllocation<Self::Page>, Self::Error> {
        if let Some(page) = self.recycled.pop() {
            return Ok(PageAllocation {
                page,
                recycled: true,
            });
        }
        let page = BenchPage(self.next_page);
        self.next_page += 1;
        Ok(PageAllocation {
            page,
            recycled: false,
        })
    }

    fn rollback_page(&mut self, page: Self::Page, _context: &mut Self::Context<'_>) {
        self.recycled.push(page);
    }

    fn prepare_append(
        &mut self,
        pages: &[BackendAppendPage<'_, Self::Page>],
        start_position: usize,
        _context: &mut Self::Context<'_>,
    ) -> Result<Self::AppendTransaction, Self::Error> {
        Ok((
            start_position,
            pages.iter().map(BackendAppendPage::rows).sum(),
        ))
    }

    fn abort_append(
        &mut self,
        _transaction: &mut Self::AppendTransaction,
        restored_pages: &[&Self::Page],
        released_pages: &[&Self::Page],
        restored_position: usize,
        context: &mut Self::Context<'_>,
    ) -> Result<(), Self::Error> {
        context.table.clear();
        context
            .table
            .extend(restored_pages.iter().map(|page| page.0));
        context.position = restored_position;
        self.recycled
            .extend(released_pages.iter().map(|page| **page));
        Ok(())
    }

    fn copy_partial_page(
        &mut self,
        _source: &Self::Page,
        _valid_tokens: usize,
        context: &mut Self::Context<'_>,
    ) -> Result<PageAllocation<Self::Page>, Self::Error> {
        self.allocate_page(context)
    }

    fn commit_append(
        &mut self,
        _transaction: &mut Self::AppendTransaction,
        commit: BackendAppendCommit<'_, Self::Page>,
        context: &mut Self::Context<'_>,
    ) -> Result<(), Self::Error> {
        context.table.clear();
        context
            .table
            .extend(commit.committed_pages().iter().map(|page| page.0));
        context.position = commit.position();
        self.recycled
            .extend(commit.released_pages().iter().map(|page| **page));
        Ok(())
    }

    fn update_page_table(
        &mut self,
        pages: &[&Self::Page],
        position: usize,
        context: &mut Self::Context<'_>,
    ) -> Result<(), Self::Error> {
        context.table.clear();
        context.table.extend(pages.iter().map(|page| page.0));
        context.position = position;
        Ok(())
    }

    fn retire_pages(
        &mut self,
        pages: Vec<Self::Page>,
        _context: &mut Self::Context<'_>,
    ) -> Result<RetireOutcome, RetireError<Self::Error, Self::Page>> {
        self.recycled.extend(pages);
        Ok(RetireOutcome::default())
    }

    fn retirement_is_immediate(&self) -> bool {
        true
    }

    fn poll_reclaimed(&mut self, _context: &mut Self::Context<'_>) -> Result<usize, Self::Error> {
        Ok(0)
    }
}

type BenchCache = SequenceCache<BenchBackend, ()>;

fn new_cache() -> BenchCache {
    new_cache_with_managed_bytes(MANAGED_BYTES)
}

fn new_cache_with_managed_bytes(max_managed_bytes: usize) -> BenchCache {
    SequenceCache::new(
        CacheConfig {
            page_tokens: PAGE_TOKENS,
            max_managed_bytes,
            max_snapshot_bytes: 0,
            max_prefix_entries: None,
            emergency_bytes: 0,
        },
        BenchBackend::default(),
    )
    .expect("valid benchmark cache")
}

fn request(max_position: usize) -> AdmissionRequest {
    AdmissionRequest {
        max_position,
        private_state_bytes: 0,
        page_table_bytes: max_position.div_ceil(PAGE_TOKENS) * size_of::<u32>(),
        allow_emergency: false,
    }
}

fn request_without_page_table(max_position: usize) -> AdmissionRequest {
    AdmissionRequest {
        max_position,
        private_state_bytes: 0,
        page_table_bytes: 0,
        allow_emergency: false,
    }
}

fn admit(
    cache: &mut BenchCache,
    prefix: Option<PrefixMatch>,
    max_position: usize,
    context: &mut BenchBackendContext,
) -> SequenceId {
    match cache
        .admit(prefix, request(max_position), context, |_, _| Ok(()))
        .expect("benchmark admission")
    {
        AdmissionOutcome::Admitted(sequence) => sequence,
        AdmissionOutcome::WouldBlock => panic!("benchmark cache unexpectedly exhausted"),
    }
}

fn inspect_reservation(
    cache: &mut BenchCache,
    reservation: &seqcache::AppendReservation,
    expected_rows: usize,
) {
    let rows = cache
        .with_append_pages(reservation, |_backend, pages| {
            Ok(pages
                .iter()
                .map(|page| page.segment().rows())
                .sum::<usize>())
        })
        .expect("inspect append pages");
    assert_eq!(rows, expected_rows);
    black_box(rows);
}

fn append_exact(
    cache: &mut BenchCache,
    sequence: SequenceId,
    rows: usize,
    context: &mut BenchBackendContext,
) {
    let reservation = cache
        .reserve_append(sequence, rows, context)
        .expect("reserve append");
    inspect_reservation(cache, &reservation, rows);
    cache
        .commit_append(reservation, rows, context)
        .expect("commit append");
}

fn operation_metrics(rows: usize, pages: usize, operations: usize) -> BenchSampleResult {
    BenchSampleResult::operations(operations as u64)
        .push_metric(MetricValue::integer("rows", rows as i64, "tokens"))
        .push_metric(MetricValue::integer("pages", pages as i64, "pages"))
}

fn count_metrics(items: usize, operations: usize, name: &'static str) -> BenchSampleResult {
    BenchSampleResult::operations(operations as u64).push_metric(MetricValue::integer(
        name,
        items as i64,
        name,
    ))
}

struct AbortBench<const START: usize, const ROWS: usize> {
    cache: BenchCache,
    context: BenchBackendContext,
    sequence: SequenceId,
}

impl<const START: usize, const ROWS: usize> BenchContext for AbortBench<START, ROWS> {
    fn prepare(_num_chunks: usize) -> Self {
        let mut cache = new_cache();
        let mut context = BenchBackendContext::default();
        let sequence = admit(&mut cache, None, START + ROWS, &mut context);
        if START != 0 {
            append_exact(&mut cache, sequence, START, &mut context);
        }
        cache.validate().expect("valid abort fixture");
        Self {
            cache,
            context,
            sequence,
        }
    }

    fn chunk_size() -> Option<usize> {
        None
    }
}

fn reserve_abort_sample<const START: usize, const ROWS: usize>(
    bench: &mut AbortBench<START, ROWS>,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    for _ in 0..chunk_size {
        let reservation = bench
            .cache
            .reserve_append(bench.sequence, ROWS, &mut bench.context)
            .expect("reserve benchmark append");
        inspect_reservation(&mut bench.cache, &reservation, ROWS);
        bench
            .cache
            .abort_append(reservation, &mut bench.context)
            .expect("abort benchmark append");
    }
    let touched_pages = (START % PAGE_TOKENS + ROWS).div_ceil(PAGE_TOKENS);
    operation_metrics(ROWS, touched_pages, chunk_size)
}

struct LifecycleBench<const RESERVED: usize, const COMMITTED: usize> {
    cache: BenchCache,
    context: BenchBackendContext,
}

impl<const RESERVED: usize, const COMMITTED: usize> BenchContext
    for LifecycleBench<RESERVED, COMMITTED>
{
    fn prepare(_num_chunks: usize) -> Self {
        assert!(COMMITTED > 0 && COMMITTED <= RESERVED);
        let mut bench = Self {
            cache: new_cache(),
            context: BenchBackendContext::default(),
        };
        run_lifecycle::<RESERVED, COMMITTED>(&mut bench);
        bench.cache.validate().expect("valid lifecycle fixture");
        bench
    }

    fn chunk_size() -> Option<usize> {
        None
    }
}

fn run_lifecycle<const RESERVED: usize, const COMMITTED: usize>(
    bench: &mut LifecycleBench<RESERVED, COMMITTED>,
) {
    let sequence = admit(&mut bench.cache, None, RESERVED, &mut bench.context);
    let reservation = bench
        .cache
        .reserve_append(sequence, RESERVED, &mut bench.context)
        .expect("reserve lifecycle append");
    inspect_reservation(&mut bench.cache, &reservation, RESERVED);
    bench
        .cache
        .commit_append(reservation, COMMITTED, &mut bench.context)
        .expect("commit lifecycle append");
    bench
        .cache
        .finish(sequence, &mut bench.context)
        .expect("finish lifecycle sequence");
}

fn lifecycle_sample<const RESERVED: usize, const COMMITTED: usize>(
    bench: &mut LifecycleBench<RESERVED, COMMITTED>,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    for _ in 0..chunk_size {
        run_lifecycle::<RESERVED, COMMITTED>(bench);
    }
    operation_metrics(RESERVED, RESERVED.div_ceil(PAGE_TOKENS), chunk_size).push_metric(
        MetricValue::integer("committed_rows", COMMITTED as i64, "tokens"),
    )
}

struct PrefixBench {
    cache: BenchCache,
    context: BenchBackendContext,
    hit_tokens: Vec<u32>,
    early_miss_tokens: Vec<u32>,
    miss_tokens: Vec<u32>,
}

impl BenchContext for PrefixBench {
    fn prepare(_num_chunks: usize) -> Self {
        let mut cache = new_cache();
        let mut context = BenchBackendContext::default();
        let sequence = admit(&mut cache, None, OBSERVED_PROMPT_TOKENS, &mut context);
        append_exact(&mut cache, sequence, OBSERVED_PREFIX_TOKENS, &mut context);
        let hit_tokens = (0..OBSERVED_PROMPT_TOKENS as u32).collect::<Vec<_>>();
        assert!(matches!(
            cache
                .retain_prefix(sequence, &hit_tokens, (), &mut context)
                .expect("retain benchmark prefix"),
            RetainOutcome::Inserted(_)
        ));
        cache
            .finish(sequence, &mut context)
            .expect("finish prefix source");
        let mut early_miss_tokens = hit_tokens.clone();
        early_miss_tokens[0] ^= u32::MAX;
        let mut miss_tokens = hit_tokens.clone();
        miss_tokens[OBSERVED_PREFIX_TOKENS - 1] ^= u32::MAX;
        assert!(cache.lookup_prefix(&hit_tokens).is_some());
        assert!(cache.lookup_prefix(&early_miss_tokens).is_none());
        assert!(cache.lookup_prefix(&miss_tokens).is_none());
        cache.validate().expect("valid prefix fixture");
        Self {
            cache,
            context,
            hit_tokens,
            early_miss_tokens,
            miss_tokens,
        }
    }

    fn chunk_size() -> Option<usize> {
        None
    }
}

fn prefix_early_miss_sample(
    bench: &mut PrefixBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    for _ in 0..chunk_size {
        assert!(
            bench
                .cache
                .lookup_prefix(black_box(&bench.early_miss_tokens))
                .is_none()
        );
    }
    operation_metrics(
        OBSERVED_PREFIX_TOKENS,
        OBSERVED_PREFIX_TOKENS / PAGE_TOKENS,
        chunk_size,
    )
}

fn prefix_hit_sample(
    bench: &mut PrefixBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    for _ in 0..chunk_size {
        black_box(
            bench
                .cache
                .lookup_prefix(black_box(&bench.hit_tokens))
                .expect("retained prefix hit"),
        );
    }
    operation_metrics(
        OBSERVED_PREFIX_TOKENS,
        OBSERVED_PREFIX_TOKENS / PAGE_TOKENS,
        chunk_size,
    )
}

fn prefix_miss_sample(
    bench: &mut PrefixBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    for _ in 0..chunk_size {
        assert!(
            bench
                .cache
                .lookup_prefix(black_box(&bench.miss_tokens))
                .is_none()
        );
    }
    operation_metrics(
        OBSERVED_PREFIX_TOKENS,
        OBSERVED_PREFIX_TOKENS / PAGE_TOKENS,
        chunk_size,
    )
}

fn prefix_restore_sample(
    bench: &mut PrefixBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    for _ in 0..chunk_size {
        let prefix = bench
            .cache
            .lookup_prefix(black_box(&bench.hit_tokens))
            .expect("retained prefix hit");
        let sequence = admit(
            &mut bench.cache,
            Some(prefix),
            OBSERVED_PROMPT_TOKENS,
            &mut bench.context,
        );
        bench
            .cache
            .finish(sequence, &mut bench.context)
            .expect("finish restored sequence");
    }
    operation_metrics(
        OBSERVED_PREFIX_TOKENS,
        OBSERVED_PREFIX_TOKENS / PAGE_TOKENS,
        chunk_size,
    )
}

struct BranchBench {
    cache: BenchCache,
    context: BenchBackendContext,
    source: SequenceId,
}

impl BenchContext for BranchBench {
    fn prepare(_num_chunks: usize) -> Self {
        let mut cache = new_cache();
        let mut context = BenchBackendContext::default();
        let source = admit(&mut cache, None, 2 * PAGE_TOKENS, &mut context);
        append_exact(&mut cache, source, PAGE_TOKENS + 64, &mut context);
        cache.validate().expect("valid branch fixture");
        Self {
            cache,
            context,
            source,
        }
    }

    fn chunk_size() -> Option<usize> {
        None
    }
}

fn branch_sample(
    bench: &mut BranchBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    for _ in 0..chunk_size {
        let branch = match bench
            .cache
            .branch(bench.source, request(2 * PAGE_TOKENS), &mut bench.context)
            .expect("branch sequence")
        {
            AdmissionOutcome::Admitted(sequence) => sequence,
            AdmissionOutcome::WouldBlock => panic!("branch unexpectedly blocked"),
        };
        bench
            .cache
            .finish(branch, &mut bench.context)
            .expect("finish branch");
    }
    operation_metrics(PAGE_TOKENS + 64, 2, chunk_size)
}

struct ConstructionBench;

impl BenchContext for ConstructionBench {
    fn prepare(_chunk_size: usize) -> Self {
        Self
    }
}

fn cache_construction_sample(
    _bench: &mut ConstructionBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    for _ in 0..chunk_size {
        black_box(new_cache());
    }
    count_metrics(1, chunk_size, "caches")
}

struct ManagerReadBench {
    cache: BenchCache,
    sequence: SequenceId,
    page: PageId,
    tokens: Vec<u32>,
}

impl BenchContext for ManagerReadBench {
    fn prepare(_chunk_size: usize) -> Self {
        let mut cache = new_cache();
        let mut context = BenchBackendContext::default();
        let sequence = admit(&mut cache, None, OBSERVED_PREFIX_TOKENS, &mut context);
        append_exact(&mut cache, sequence, OBSERVED_PREFIX_TOKENS, &mut context);
        let tokens = (0..OBSERVED_PREFIX_TOKENS as u32).collect::<Vec<_>>();
        assert!(matches!(
            cache
                .retain_prefix(sequence, &tokens, (), &mut context)
                .expect("retain manager-read prefix"),
            RetainOutcome::Inserted(_)
        ));
        let page = *cache
            .page_table(sequence)
            .expect("manager-read page table")
            .pages()
            .last()
            .expect("manager-read fixture has pages");
        cache.validate().expect("valid manager-read fixture");
        Self {
            cache,
            sequence,
            page,
            tokens,
        }
    }
}

fn stats_snapshot_sample(
    bench: &mut ManagerReadBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    for _ in 0..chunk_size {
        black_box(bench.cache.stats());
    }
    count_metrics(1, chunk_size, "snapshots")
}

fn cacheable_position_sample(
    bench: &mut ManagerReadBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    for _ in 0..chunk_size {
        black_box(
            bench
                .cache
                .cacheable_prefix_tokens(black_box(OBSERVED_PROMPT_TOKENS)),
        );
    }
    count_metrics(1, chunk_size, "positions")
}

fn page_table_view_sample(
    bench: &mut ManagerReadBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    for _ in 0..chunk_size {
        let table = bench
            .cache
            .page_table(black_box(bench.sequence))
            .expect("read page table");
        black_box((table.position(), table.pages().len()));
    }
    operation_metrics(
        OBSERVED_PREFIX_TOKENS,
        OBSERVED_PREFIX_TOKENS / PAGE_TOKENS,
        chunk_size,
    )
}

fn page_handle_resolution_sample(
    bench: &mut ManagerReadBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    for _ in 0..chunk_size {
        black_box(
            bench
                .cache
                .page(black_box(bench.page))
                .expect("resolve page handle"),
        );
    }
    count_metrics(1, chunk_size, "pages")
}

fn contains_prefix_sample(
    bench: &mut ManagerReadBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    for _ in 0..chunk_size {
        assert!(
            bench
                .cache
                .contains_prefix(black_box(&bench.tokens), black_box(OBSERVED_PREFIX_TOKENS),)
        );
    }
    operation_metrics(
        OBSERVED_PREFIX_TOKENS,
        OBSERVED_PREFIX_TOKENS / PAGE_TOKENS,
        chunk_size,
    )
}

fn retained_state_validation_sample(
    bench: &mut ManagerReadBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    for _ in 0..chunk_size {
        bench.cache.validate().expect("validate retained state");
    }
    operation_metrics(
        OBSERVED_PREFIX_TOKENS,
        OBSERVED_PREFIX_TOKENS / PAGE_TOKENS,
        chunk_size,
    )
}

struct AppendLeaseBench<const START: usize, const ROWS: usize> {
    cache: BenchCache,
    reservation: AppendReservation,
}

impl<const START: usize, const ROWS: usize> BenchContext for AppendLeaseBench<START, ROWS> {
    fn prepare(_chunk_size: usize) -> Self {
        let mut cache = new_cache();
        let mut context = BenchBackendContext::default();
        let sequence = admit(&mut cache, None, START + ROWS, &mut context);
        if START != 0 {
            append_exact(&mut cache, sequence, START, &mut context);
        }
        let reservation = cache
            .reserve_append(sequence, ROWS, &mut context)
            .expect("reserve append lease fixture");
        cache.validate().expect("valid append lease fixture");
        Self { cache, reservation }
    }
}

fn append_lease_sample<const START: usize, const ROWS: usize>(
    bench: &mut AppendLeaseBench<START, ROWS>,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    for _ in 0..chunk_size {
        let rows = bench
            .cache
            .with_append_pages(&bench.reservation, |_backend, pages| {
                Ok(pages
                    .iter()
                    .map(|page| page.segment().rows())
                    .sum::<usize>())
            })
            .expect("borrow append lease");
        black_box(rows);
    }
    let touched_pages = (START % PAGE_TOKENS + ROWS).div_ceil(PAGE_TOKENS);
    operation_metrics(ROWS, touched_pages, chunk_size)
}

const BATCH_SEQUENCES: usize = 8;

struct BatchLeaseBench<const ROWS: usize> {
    cache: BenchCache,
    reservations: Vec<AppendReservation>,
}

impl<const ROWS: usize> BenchContext for BatchLeaseBench<ROWS> {
    fn prepare(_chunk_size: usize) -> Self {
        let mut cache = new_cache();
        let mut context = BenchBackendContext::default();
        let mut reservations = Vec::with_capacity(BATCH_SEQUENCES);
        for _ in 0..BATCH_SEQUENCES {
            let sequence = admit(&mut cache, None, ROWS, &mut context);
            reservations.push(
                cache
                    .reserve_append(sequence, ROWS, &mut context)
                    .expect("reserve batch lease fixture"),
            );
        }
        cache.validate().expect("valid batch lease fixture");
        Self {
            cache,
            reservations,
        }
    }
}

fn batch_lease_sample<const ROWS: usize>(
    bench: &mut BatchLeaseBench<ROWS>,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    for _ in 0..chunk_size {
        let rows = bench
            .cache
            .with_append_reservations(&bench.reservations, |_backend, reservations| {
                Ok(reservations
                    .iter()
                    .map(|pages| {
                        pages
                            .iter()
                            .map(|page| page.segment().rows())
                            .sum::<usize>()
                    })
                    .sum::<usize>())
            })
            .expect("borrow batched append leases");
        black_box(rows);
    }
    operation_metrics(
        BATCH_SEQUENCES * ROWS,
        BATCH_SEQUENCES * ROWS.div_ceil(PAGE_TOKENS),
        chunk_size,
    )
}

struct ShortPrefixBench {
    cache: BenchCache,
    hit_tokens: Vec<u32>,
    miss_tokens: Vec<u32>,
}

impl BenchContext for ShortPrefixBench {
    fn prepare(_chunk_size: usize) -> Self {
        let mut cache = new_cache();
        let mut context = BenchBackendContext::default();
        let sequence = admit(&mut cache, None, PAGE_TOKENS, &mut context);
        append_exact(&mut cache, sequence, PAGE_TOKENS, &mut context);
        let hit_tokens = (0..=PAGE_TOKENS as u32).collect::<Vec<_>>();
        assert!(matches!(
            cache
                .retain_prefix(sequence, &hit_tokens, (), &mut context)
                .expect("retain short prefix"),
            RetainOutcome::Inserted(_)
        ));
        cache
            .finish(sequence, &mut context)
            .expect("finish short prefix source");
        let mut miss_tokens = hit_tokens.clone();
        miss_tokens[0] ^= u32::MAX;
        cache.validate().expect("valid short prefix fixture");
        Self {
            cache,
            hit_tokens,
            miss_tokens,
        }
    }
}

fn short_prefix_hit_sample(
    bench: &mut ShortPrefixBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    for _ in 0..chunk_size {
        black_box(
            bench
                .cache
                .lookup_prefix(black_box(&bench.hit_tokens))
                .expect("short prefix hit"),
        );
    }
    operation_metrics(PAGE_TOKENS, 1, chunk_size)
}

fn short_prefix_miss_sample(
    bench: &mut ShortPrefixBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    for _ in 0..chunk_size {
        assert!(
            bench
                .cache
                .lookup_prefix(black_box(&bench.miss_tokens))
                .is_none()
        );
    }
    operation_metrics(PAGE_TOKENS, 1, chunk_size)
}

fn distinct_tokens(entry: usize, tokens: usize) -> Vec<u32> {
    let seed = (entry as u32 + 1).wrapping_mul(1_000_003);
    (0..tokens)
        .map(|offset| seed.wrapping_add(offset as u32))
        .collect()
}

struct PrefixInsertionBench<const PAGES: usize> {
    cache: BenchCache,
    context: BenchBackendContext,
    sequence: SequenceId,
    token_sets: Vec<Vec<u32>>,
}

impl<const PAGES: usize> BenchContext for PrefixInsertionBench<PAGES> {
    fn prepare(chunk_size: usize) -> Self {
        let prefix_tokens = PAGES * PAGE_TOKENS;
        let mut cache = new_cache();
        let mut context = BenchBackendContext::default();
        let sequence = admit(&mut cache, None, prefix_tokens, &mut context);
        append_exact(&mut cache, sequence, prefix_tokens, &mut context);
        let token_sets = (0..chunk_size)
            .map(|entry| distinct_tokens(entry, prefix_tokens))
            .collect();
        cache.validate().expect("valid prefix insertion fixture");
        Self {
            cache,
            context,
            sequence,
            token_sets,
        }
    }

    fn chunk_size() -> Option<usize> {
        Some(FIXED_PREFIX_MUTATION_OPERATIONS)
    }
}

fn prefix_insertion_sample<const PAGES: usize>(
    bench: &mut PrefixInsertionBench<PAGES>,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    for tokens in bench.token_sets.iter().take(chunk_size) {
        let outcome = bench
            .cache
            .retain_prefix(bench.sequence, black_box(tokens), (), &mut bench.context)
            .expect("insert fresh prefix");
        assert!(matches!(outcome, RetainOutcome::Inserted(_)));
        black_box(outcome);
    }
    operation_metrics(PAGES * PAGE_TOKENS, PAGES, chunk_size).push_metric(MetricValue::integer(
        "resulting_entries",
        chunk_size as i64,
        "entries",
    ))
}

struct PrefixEvictionBench<const PAGES: usize> {
    cache: BenchCache,
    context: BenchBackendContext,
    entries: Vec<PrefixEntryId>,
}

impl<const PAGES: usize> BenchContext for PrefixEvictionBench<PAGES> {
    fn prepare(chunk_size: usize) -> Self {
        let prefix_tokens = PAGES * PAGE_TOKENS;
        let mut cache = new_cache();
        let mut context = BenchBackendContext::default();
        let sequence = admit(&mut cache, None, prefix_tokens, &mut context);
        append_exact(&mut cache, sequence, prefix_tokens, &mut context);
        let mut entries = Vec::with_capacity(chunk_size);
        for entry in 0..chunk_size {
            let tokens = distinct_tokens(entry, prefix_tokens);
            let RetainOutcome::Inserted(id) = cache
                .retain_prefix(sequence, &tokens, (), &mut context)
                .expect("prepare retained prefix")
            else {
                panic!("fresh token set must insert a prefix");
            };
            entries.push(id);
        }
        cache.validate().expect("valid prefix eviction fixture");
        Self {
            cache,
            context,
            entries,
        }
    }

    fn chunk_size() -> Option<usize> {
        Some(FIXED_PREFIX_MUTATION_OPERATIONS)
    }
}

fn prefix_eviction_sample<const PAGES: usize>(
    bench: &mut PrefixEvictionBench<PAGES>,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    for entry in bench.entries.iter().copied().take(chunk_size) {
        bench
            .cache
            .evict_prefix(entry, &mut bench.context)
            .expect("evict retained prefix");
    }
    operation_metrics(PAGES * PAGE_TOKENS, PAGES, chunk_size).push_metric(MetricValue::integer(
        "starting_entries",
        chunk_size as i64,
        "entries",
    ))
}

struct DuplicateRetentionBench {
    cache: BenchCache,
    context: BenchBackendContext,
    sequence: SequenceId,
    tokens: Vec<u32>,
}

impl BenchContext for DuplicateRetentionBench {
    fn prepare(_chunk_size: usize) -> Self {
        let mut cache = new_cache();
        let mut context = BenchBackendContext::default();
        let sequence = admit(&mut cache, None, OBSERVED_PREFIX_TOKENS, &mut context);
        append_exact(&mut cache, sequence, OBSERVED_PREFIX_TOKENS, &mut context);
        let tokens = distinct_tokens(0, OBSERVED_PREFIX_TOKENS);
        cache
            .retain_prefix(sequence, &tokens, (), &mut context)
            .expect("retain duplicate fixture");
        Self {
            cache,
            context,
            sequence,
            tokens,
        }
    }
}

fn duplicate_retention_sample(
    bench: &mut DuplicateRetentionBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    for _ in 0..chunk_size {
        let outcome = bench
            .cache
            .retain_prefix(
                bench.sequence,
                black_box(&bench.tokens),
                (),
                &mut bench.context,
            )
            .expect("retain duplicate prefix");
        assert!(matches!(outcome, RetainOutcome::Duplicate(_)));
        black_box(outcome);
    }
    operation_metrics(
        OBSERVED_PREFIX_TOKENS,
        OBSERVED_PREFIX_TOKENS / PAGE_TOKENS,
        chunk_size,
    )
}

struct AdmissionBench<const MAX_POSITION: usize> {
    cache: BenchCache,
    context: BenchBackendContext,
}

impl<const MAX_POSITION: usize> BenchContext for AdmissionBench<MAX_POSITION> {
    fn prepare(_chunk_size: usize) -> Self {
        Self {
            cache: new_cache(),
            context: BenchBackendContext::default(),
        }
    }
}

fn admission_sample<const MAX_POSITION: usize>(
    bench: &mut AdmissionBench<MAX_POSITION>,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    for _ in 0..chunk_size {
        let sequence = admit(&mut bench.cache, None, MAX_POSITION, &mut bench.context);
        bench
            .cache
            .finish(sequence, &mut bench.context)
            .expect("cancel admitted benchmark sequence");
    }
    operation_metrics(MAX_POSITION, MAX_POSITION.div_ceil(PAGE_TOKENS), chunk_size)
}

struct AdmissionPressureBench {
    cache: BenchCache,
    context: BenchBackendContext,
}

impl BenchContext for AdmissionPressureBench {
    fn prepare(_chunk_size: usize) -> Self {
        let mut cache = new_cache_with_managed_bytes(PAGE_BYTES + size_of::<u32>());
        let mut context = BenchBackendContext::default();
        black_box(admit(&mut cache, None, PAGE_TOKENS, &mut context));
        Self { cache, context }
    }
}

fn admission_would_block_sample(
    bench: &mut AdmissionPressureBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    for _ in 0..chunk_size {
        let outcome = bench
            .cache
            .admit(
                None,
                request(PAGE_TOKENS),
                &mut bench.context,
                |_, _| Ok(()),
            )
            .expect("admission pressure is not an error");
        assert_eq!(outcome, AdmissionOutcome::WouldBlock);
        black_box(outcome);
    }
    operation_metrics(PAGE_TOKENS, 1, chunk_size)
}

struct AdmissionEvictionBench {
    cache: BenchCache,
    context: BenchBackendContext,
}

impl BenchContext for AdmissionEvictionBench {
    fn prepare(chunk_size: usize) -> Self {
        let mut cache = new_cache_with_managed_bytes(chunk_size * PAGE_BYTES);
        let mut context = BenchBackendContext::default();
        for entry in 0..chunk_size {
            let sequence = match cache
                .admit(
                    None,
                    request_without_page_table(PAGE_TOKENS),
                    &mut context,
                    |_, _| Ok(()),
                )
                .expect("admit prefix source")
            {
                AdmissionOutcome::Admitted(sequence) => sequence,
                AdmissionOutcome::WouldBlock => panic!("prefix source must fit"),
            };
            append_exact(&mut cache, sequence, PAGE_TOKENS, &mut context);
            let tokens = distinct_tokens(entry, PAGE_TOKENS);
            assert!(matches!(
                cache
                    .retain_prefix(sequence, &tokens, (), &mut context)
                    .expect("retain admission-pressure prefix"),
                RetainOutcome::Inserted(_)
            ));
            cache
                .finish(sequence, &mut context)
                .expect("finish admission-pressure prefix source");
        }
        assert_eq!(cache.stats().retained_prefix_entries, chunk_size);
        cache.validate().expect("valid admission eviction fixture");
        Self { cache, context }
    }

    fn chunk_size() -> Option<usize> {
        Some(FIXED_PREFIX_MUTATION_OPERATIONS)
    }
}

fn admission_with_lru_eviction_sample(
    bench: &mut AdmissionEvictionBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    for _ in 0..chunk_size {
        let outcome = bench
            .cache
            .admit(
                None,
                request_without_page_table(PAGE_TOKENS),
                &mut bench.context,
                |_, _| Ok(()),
            )
            .expect("admission with LRU eviction");
        let AdmissionOutcome::Admitted(sequence) = outcome else {
            panic!("one evicted prefix page must make room");
        };
        black_box(sequence);
    }
    operation_metrics(PAGE_TOKENS, 1, chunk_size).push_metric(MetricValue::integer(
        "starting_entries",
        chunk_size as i64,
        "entries",
    ))
}

struct FinishBench<const PAGES: usize> {
    cache: BenchCache,
    context: BenchBackendContext,
    sequences: Vec<SequenceId>,
}

impl<const PAGES: usize> BenchContext for FinishBench<PAGES> {
    fn prepare(chunk_size: usize) -> Self {
        let rows = PAGES * PAGE_TOKENS;
        let mut cache = new_cache();
        let mut context = BenchBackendContext::default();
        let mut sequences = Vec::with_capacity(chunk_size);
        for _ in 0..chunk_size {
            let sequence = admit(&mut cache, None, rows, &mut context);
            append_exact(&mut cache, sequence, rows, &mut context);
            sequences.push(sequence);
        }
        cache.validate().expect("valid finish fixture");
        Self {
            cache,
            context,
            sequences,
        }
    }

    fn chunk_size() -> Option<usize> {
        Some(FIXED_FINISH_OPERATIONS)
    }
}

fn finish_sample<const PAGES: usize>(
    bench: &mut FinishBench<PAGES>,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    for sequence in bench.sequences.iter().copied().take(chunk_size) {
        bench
            .cache
            .finish(sequence, &mut bench.context)
            .expect("finish resident sequence");
    }
    operation_metrics(PAGES * PAGE_TOKENS, PAGES, chunk_size).push_metric(MetricValue::integer(
        "starting_sequences",
        chunk_size as i64,
        "sequences",
    ))
}

struct ReservedFinishBench<const PAGES: usize> {
    cache: BenchCache,
    context: BenchBackendContext,
    sequences: Vec<SequenceId>,
}

impl<const PAGES: usize> BenchContext for ReservedFinishBench<PAGES> {
    fn prepare(chunk_size: usize) -> Self {
        let mut cache = new_cache();
        let mut context = BenchBackendContext::default();
        let sequences = (0..chunk_size)
            .map(|_| admit(&mut cache, None, PAGES * PAGE_TOKENS, &mut context))
            .collect();
        cache.validate().expect("valid reserved finish fixture");
        Self {
            cache,
            context,
            sequences,
        }
    }

    fn chunk_size() -> Option<usize> {
        Some(FIXED_FINISH_OPERATIONS)
    }
}

fn reserved_finish_sample<const PAGES: usize>(
    bench: &mut ReservedFinishBench<PAGES>,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    for sequence in bench.sequences.iter().copied().take(chunk_size) {
        bench
            .cache
            .finish(sequence, &mut bench.context)
            .expect("finish reserved sequence");
    }
    operation_metrics(PAGES * PAGE_TOKENS, PAGES, chunk_size).push_metric(MetricValue::integer(
        "starting_sequences",
        chunk_size as i64,
        "sequences",
    ))
}

struct ReclaimPollBench<const ACTIVE_SEQUENCES: usize> {
    cache: BenchCache,
    context: BenchBackendContext,
}

impl<const ACTIVE_SEQUENCES: usize> BenchContext for ReclaimPollBench<ACTIVE_SEQUENCES> {
    fn prepare(_chunk_size: usize) -> Self {
        let mut cache = new_cache();
        let mut context = BenchBackendContext::default();
        for _ in 0..ACTIVE_SEQUENCES {
            black_box(admit(&mut cache, None, PAGE_TOKENS, &mut context));
        }
        cache.validate().expect("valid reclaim polling fixture");
        Self { cache, context }
    }
}

fn reclaim_poll_sample<const ACTIVE_SEQUENCES: usize>(
    bench: &mut ReclaimPollBench<ACTIVE_SEQUENCES>,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    for _ in 0..chunk_size {
        black_box(
            bench
                .cache
                .reclaim_deferred(&mut bench.context)
                .expect("poll deferred reclamation"),
        );
    }
    count_metrics(ACTIVE_SEQUENCES, chunk_size, "sequences")
}

struct EmptyValidationBench {
    cache: BenchCache,
}

impl BenchContext for EmptyValidationBench {
    fn prepare(_chunk_size: usize) -> Self {
        Self { cache: new_cache() }
    }
}

fn empty_validation_sample(
    bench: &mut EmptyValidationBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    for _ in 0..chunk_size {
        bench.cache.validate().expect("validate empty cache");
    }
    count_metrics(0, chunk_size, "sequences")
}

const VALIDATION_SEQUENCES: usize = 64;
const VALIDATION_PAGES: usize = 4;

struct ActiveValidationBench {
    cache: BenchCache,
}

impl BenchContext for ActiveValidationBench {
    fn prepare(_chunk_size: usize) -> Self {
        let mut cache = new_cache();
        let mut context = BenchBackendContext::default();
        for _ in 0..VALIDATION_SEQUENCES {
            let sequence = admit(
                &mut cache,
                None,
                VALIDATION_PAGES * PAGE_TOKENS,
                &mut context,
            );
            append_exact(
                &mut cache,
                sequence,
                VALIDATION_PAGES * PAGE_TOKENS,
                &mut context,
            );
        }
        cache.validate().expect("valid active validation fixture");
        Self { cache }
    }
}

fn active_validation_sample(
    bench: &mut ActiveValidationBench,
    chunk_size: usize,
    _chunk_num: usize,
) -> BenchSampleResult {
    for _ in 0..chunk_size {
        bench.cache.validate().expect("validate active cache");
    }
    BenchSampleResult::operations(chunk_size as u64)
        .push_metric(MetricValue::integer(
            "sequences",
            VALIDATION_SEQUENCES as i64,
            "sequences",
        ))
        .push_metric(MetricValue::integer(
            "pages",
            (VALIDATION_SEQUENCES * VALIDATION_PAGES) as i64,
            "pages",
        ))
}

fn main() {
    let options = BenchmarkMainOptions {
        suite: Some("seqcache".to_string()),
        comparison_policy: ComparisonPolicy::None,
        save_results: false,
        runtime: BenchmarkRuntimeOptions {
            warm_up_duration: Duration::from_millis(100),
            benchmark_duration: Duration::from_millis(500),
            min_samples: 5,
            max_samples: 10,
        },
        ..BenchmarkMainOptions::default()
    };

    run_benchmark_main(options, |runner| {
        runner.group::<ConstructionBench>("manager construction", |group| {
            group
                .throughput(Throughput::per_operation(1, "caches"))
                .bench_sample("empty_cache", cache_construction_sample);
        });
        runner.group::<ManagerReadBench>("manager read primitives", |group| {
            group
                .throughput(Throughput::per_operation(1, "snapshots"))
                .bench_sample("stats_snapshot", stats_snapshot_sample);
            group
                .throughput(Throughput::per_operation(1, "positions"))
                .bench_sample("cacheable_prefix_position", cacheable_position_sample);
            group
                .throughput(Throughput::per_operation(1, "tables"))
                .bench_sample("page_table_view_37_pages", page_table_view_sample);
            group
                .throughput(Throughput::per_operation(1, "pages"))
                .bench_sample("resolve_last_page_handle", page_handle_resolution_sample);
            group
                .throughput(Throughput::per_operation(
                    OBSERVED_PREFIX_TOKENS as u64,
                    "prefix tokens",
                ))
                .bench_sample("contains_exact_prefix_37_pages", contains_prefix_sample);
        });

        runner.group::<AppendLeaseBench<0, 128>>("append reservation access", |group| {
            group
                .throughput(Throughput::per_operation(128, "tokens"))
                .bench_sample("borrow_1_page", append_lease_sample::<0, 128>);
        });
        runner.group::<AppendLeaseBench<64, 2048>>("append reservation access", |group| {
            group
                .throughput(Throughput::per_operation(2_048, "tokens"))
                .bench_sample(
                    "borrow_partial_tail_17_segments",
                    append_lease_sample::<64, 2048>,
                );
        });
        runner.group::<BatchLeaseBench<2048>>("append reservation access", |group| {
            group
                .throughput(Throughput::per_operation(
                    (BATCH_SEQUENCES * 2_048) as u64,
                    "tokens",
                ))
                .bench_sample("borrow_batch_8_by_16_pages", batch_lease_sample::<2048>);
        });

        runner.group::<AbortBench<0, 128>>("append transaction reserve and abort", |group| {
            group
                .throughput(Throughput::per_operation(128, "tokens"))
                .bench_sample("boundary_128_rows_1_page", reserve_abort_sample::<0, 128>);
        });
        runner.group::<AbortBench<0, 2048>>("append transaction reserve and abort", |group| {
            group
                .throughput(Throughput::per_operation(2_048, "tokens"))
                .bench_sample(
                    "boundary_2048_rows_16_pages",
                    reserve_abort_sample::<0, 2048>,
                );
        });
        runner.group::<AbortBench<0, 8192>>("append transaction reserve and abort", |group| {
            group
                .throughput(Throughput::per_operation(8_192, "tokens"))
                .bench_sample(
                    "boundary_8192_rows_64_pages",
                    reserve_abort_sample::<0, 8192>,
                );
        });
        runner.group::<AbortBench<64, 2048>>("append transaction reserve and abort", |group| {
            group
                .throughput(Throughput::per_operation(2_048, "tokens"))
                .bench_sample(
                    "partial_tail_64_plus_2048_rows_17_segments",
                    reserve_abort_sample::<64, 2048>,
                );
        });

        runner.group::<LifecycleBench<128, 128>>("complete append lifecycle", |group| {
            group
                .throughput(Throughput::per_operation(128, "tokens"))
                .bench_sample("exact_128_rows", lifecycle_sample::<128, 128>);
        });
        runner.group::<LifecycleBench<2048, 2048>>("complete append lifecycle", |group| {
            group
                .throughput(Throughput::per_operation(2_048, "tokens"))
                .bench_sample("exact_2048_rows", lifecycle_sample::<2048, 2048>);
        });
        runner.group::<LifecycleBench<8192, 8192>>("complete append lifecycle", |group| {
            group
                .throughput(Throughput::per_operation(8_192, "tokens"))
                .bench_sample("exact_8192_rows", lifecycle_sample::<8192, 8192>);
        });
        runner.group::<LifecycleBench<2048, 640>>("complete append lifecycle", |group| {
            group
                .throughput(Throughput::per_operation(2_048, "reserved tokens"))
                .bench_sample(
                    "partial_commit_640_of_2048_rows",
                    lifecycle_sample::<2048, 640>,
                );
        });

        runner.group::<AdmissionBench<128>>("strict admission", |group| {
            group
                .throughput(Throughput::per_operation(1, "admissions"))
                .bench_sample("cold_admit_cancel_1_page", admission_sample::<128>);
        });
        runner.group::<AdmissionBench<2048>>("strict admission", |group| {
            group
                .throughput(Throughput::per_operation(1, "admissions"))
                .bench_sample("cold_admit_cancel_16_pages", admission_sample::<2048>);
        });
        runner.group::<AdmissionPressureBench>("strict admission", |group| {
            group
                .throughput(Throughput::per_operation(1, "decisions"))
                .bench_sample("would_block_at_capacity", admission_would_block_sample);
        });
        runner.group::<AdmissionEvictionBench>("strict admission", |group| {
            group
                .throughput(Throughput::per_operation(1, "admissions"))
                .bench_sample(
                    "admit_with_lru_eviction_from_128_entries",
                    admission_with_lru_eviction_sample,
                );
        });

        runner.group::<ReservedFinishBench<16>>("sequence finish", |group| {
            group
                .throughput(Throughput::per_operation(1, "sequences"))
                .bench_sample("release_16_reserved_pages", reserved_finish_sample::<16>);
        });
        runner.group::<FinishBench<1>>("sequence finish", |group| {
            group
                .throughput(Throughput::per_operation(1, "sequences"))
                .bench_sample("retire_1_resident_page", finish_sample::<1>);
        });
        runner.group::<FinishBench<16>>("sequence finish", |group| {
            group
                .throughput(Throughput::per_operation(1, "sequences"))
                .bench_sample("retire_16_resident_pages", finish_sample::<16>);
        });

        runner.group::<ShortPrefixBench>("retained prefix lookup scaling", |group| {
            group
                .throughput(Throughput::per_operation(
                    PAGE_TOKENS as u64,
                    "prefix tokens",
                ))
                .bench_sample("hit_1_page", short_prefix_hit_sample);
            group
                .throughput(Throughput::per_operation(
                    PAGE_TOKENS as u64,
                    "prefix tokens",
                ))
                .bench_sample("miss_first_block_1_page", short_prefix_miss_sample);
        });
        runner.group::<PrefixBench>("retained prefix index", |group| {
            group
                .throughput(Throughput::per_operation(
                    OBSERVED_PREFIX_TOKENS as u64,
                    "prefix tokens",
                ))
                .bench_sample("hit_4736_of_5433_tokens", prefix_hit_sample);
            group
                .throughput(Throughput::per_operation(
                    OBSERVED_PREFIX_TOKENS as u64,
                    "prefix tokens",
                ))
                .bench_sample("miss_first_token", prefix_early_miss_sample);
            group
                .throughput(Throughput::per_operation(
                    OBSERVED_PREFIX_TOKENS as u64,
                    "prefix tokens",
                ))
                .bench_sample("miss_last_prefix_token", prefix_miss_sample);
        });
        runner.group::<PrefixBench>("retained prefix restore lifecycle", |group| {
            group
                .throughput(Throughput::per_operation(
                    OBSERVED_PREFIX_TOKENS as u64,
                    "restored tokens",
                ))
                .bench_sample("restore_4736_admit_finish", prefix_restore_sample);
        });
        runner.group::<DuplicateRetentionBench>("retained prefix mutation", |group| {
            group
                .throughput(Throughput::per_operation(1, "retentions"))
                .bench_sample("duplicate_37_pages", duplicate_retention_sample);
        });
        runner.group::<PrefixInsertionBench<1>>("retained prefix mutation", |group| {
            group
                .throughput(Throughput::per_operation(1, "insertions"))
                .bench_sample("insert_fresh_1_page", prefix_insertion_sample::<1>);
        });
        runner.group::<PrefixInsertionBench<37>>("retained prefix mutation", |group| {
            group
                .throughput(Throughput::per_operation(1, "insertions"))
                .bench_sample("insert_fresh_37_pages", prefix_insertion_sample::<37>);
        });
        runner.group::<PrefixEvictionBench<1>>("retained prefix mutation", |group| {
            group
                .throughput(Throughput::per_operation(1, "evictions"))
                .bench_sample("evict_1_page", prefix_eviction_sample::<1>);
        });
        runner.group::<PrefixEvictionBench<37>>("retained prefix mutation", |group| {
            group
                .throughput(Throughput::per_operation(1, "evictions"))
                .bench_sample("evict_37_pages", prefix_eviction_sample::<37>);
        });

        runner.group::<BranchBench>("copy-on-write branch lifecycle", |group| {
            group
                .throughput(Throughput::per_operation(1, "branches"))
                .bench_sample("one_full_page_plus_64_tail", branch_sample);
        });

        runner.group::<ReclaimPollBench<0>>("accounting refresh scaling", |group| {
            group
                .throughput(Throughput::per_operation(1, "polls"))
                .bench_sample("reclaim_poll_empty", reclaim_poll_sample::<0>);
        });
        runner.group::<ReclaimPollBench<64>>("accounting refresh scaling", |group| {
            group
                .throughput(Throughput::per_operation(1, "polls"))
                .bench_sample("reclaim_poll_64_sequences", reclaim_poll_sample::<64>);
        });
        runner.group::<ReclaimPollBench<1024>>("accounting refresh scaling", |group| {
            group
                .throughput(Throughput::per_operation(1, "polls"))
                .bench_sample("reclaim_poll_1024_sequences", reclaim_poll_sample::<1024>);
        });

        runner.group::<EmptyValidationBench>("invariant validation scaling", |group| {
            group
                .throughput(Throughput::per_operation(1, "validations"))
                .bench_sample("empty_cache", empty_validation_sample);
        });
        runner.group::<ManagerReadBench>("invariant validation scaling", |group| {
            group
                .throughput(Throughput::per_operation(1, "validations"))
                .bench_sample(
                    "one_37_page_retained_prefix",
                    retained_state_validation_sample,
                );
        });
        runner.group::<ActiveValidationBench>("invariant validation scaling", |group| {
            group
                .throughput(Throughput::per_operation(1, "validations"))
                .bench_sample("64_sequences_256_pages", active_validation_sample);
        });
    });
}
