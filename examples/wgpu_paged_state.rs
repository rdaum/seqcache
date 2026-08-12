//! Native `wgpu` backend with a storage-buffer page slab.
//!
//! The backend stores every physical page in one preallocated GPU buffer and
//! publishes per-sequence logical page tables containing physical slot indices.
//! One WGSL compute dispatch writes an append across every page in a reservation.
//!
//! This example is native-only because its conservative reclamation path uses
//! blocking [`wgpu::Device::poll`]. Browser WebGPU requires an asynchronous
//! completion strategy instead.

#[cfg(target_arch = "wasm32")]
compile_error!("the wgpu_paged_state example supports native targets only");

use std::error::Error;
use std::fmt;
use std::mem::size_of;
use std::sync::mpsc;

use seqcache::{
    AdmissionOutcome, AdmissionRequest, AppendSegment, BackendAppendCommit, BackendAppendPage,
    CacheConfig, PageAllocation, PageBackend, RetireError, RetireOutcome, SequenceCache,
    SequenceId,
};

const PAGE_TOKENS: usize = 4;
const MAX_PHYSICAL_PAGES: usize = 8;
const MAX_SEQUENCE_TOKENS: usize = 16;
const WORKGROUP_SIZE: usize = 64;

const WRITE_PAGED_ROWS_WGSL: &str = r#"
struct AppendParameters {
    start_position: u32,
    rows: u32,
    page_tokens: u32,
    _padding: u32,
}

@group(0) @binding(0)
var<storage, read_write> page_slab: array<f32>;

@group(0) @binding(1)
var<storage, read> page_table: array<u32>;

@group(0) @binding(2)
var<uniform> parameters: AppendParameters;

@compute @workgroup_size(64)
fn write_paged_rows(@builtin(global_invocation_id) invocation: vec3<u32>) {
    let input_row = invocation.x;
    if input_row >= parameters.rows {
        return;
    }

    let position = parameters.start_position + input_row;
    let logical_page = position / parameters.page_tokens;
    let page_offset = position % parameters.page_tokens;
    let physical_page = page_table[logical_page];
    let slab_offset = physical_page * parameters.page_tokens + page_offset;
    page_slab[slab_offset] = f32(position + 1u);
}
"#;

#[derive(Debug)]
enum WgpuBackendError {
    Adapter(String),
    Device(String),
    OutOfPages,
    StalePage(u32),
    InvalidGeometry(&'static str),
    Poll(String),
    Map(String),
}

impl fmt::Display for WgpuBackendError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Adapter(error) => write!(formatter, "failed to request a wgpu adapter: {error}"),
            Self::Device(error) => write!(formatter, "failed to request a wgpu device: {error}"),
            Self::OutOfPages => formatter.write_str("wgpu page slab is exhausted"),
            Self::StalePage(slot) => write!(formatter, "wgpu page slot {slot} is not live"),
            Self::InvalidGeometry(detail) => formatter.write_str(detail),
            Self::Poll(error) => write!(formatter, "wgpu submission wait failed: {error}"),
            Self::Map(error) => write!(formatter, "wgpu readback mapping failed: {error}"),
        }
    }
}

impl Error for WgpuBackendError {}

/// Opaque slot in the backend's preallocated storage buffer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct WgpuPage {
    slot: u32,
}

/// Backend-native device table plus its host shadow.
struct WgpuPageTable {
    buffer: wgpu::Buffer,
    slots: Vec<u32>,
    position: usize,
    capacity: usize,
}

impl WgpuPageTable {
    fn new(device: &wgpu::Device, max_position: usize) -> Self {
        let capacity = max_position.div_ceil(PAGE_TOKENS);
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("seqcache logical page table"),
            size: (capacity * size_of::<u32>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        Self {
            buffer,
            slots: Vec::with_capacity(capacity),
            position: 0,
            capacity,
        }
    }

    fn managed_bytes(&self) -> usize {
        // The fixed device buffer and the reserved host shadow are both
        // sequence-owned page-table state.
        (self.capacity + self.slots.capacity()) * size_of::<u32>()
    }

    fn publish(
        &mut self,
        pages: &[&WgpuPage],
        position: usize,
        queue: &wgpu::Queue,
    ) -> Result<wgpu::SubmissionIndex, WgpuBackendError> {
        if pages.len() > self.capacity {
            return Err(WgpuBackendError::InvalidGeometry(
                "logical page table exceeds its admitted capacity",
            ));
        }
        if position > pages.len() * PAGE_TOKENS {
            return Err(WgpuBackendError::InvalidGeometry(
                "logical page table is too short for its position",
            ));
        }
        let mut bytes = Vec::with_capacity(pages.len() * size_of::<u32>());
        for page in pages {
            bytes.extend_from_slice(&page.slot.to_le_bytes());
        }
        if !bytes.is_empty() {
            queue.write_buffer(&self.buffer, 0, &bytes);
        }
        // Queue writes begin before commands in the next submission. An empty
        // submission makes publication explicit even when no model dispatch
        // immediately follows.
        let submission = queue.submit([]);
        self.slots.clear();
        self.slots.extend(pages.iter().map(|page| page.slot));
        self.position = position;
        Ok(submission)
    }
}

/// Per-sequence state threaded through manager operations.
struct WgpuExecution<'a> {
    page_table: &'a mut WgpuPageTable,
}

/// Fixed-capacity native wgpu storage and compute pipeline.
struct WgpuPageBackend {
    device: wgpu::Device,
    queue: wgpu::Queue,
    adapter_name: String,
    adapter_backend: wgpu::Backend,
    page_slab: wgpu::Buffer,
    copy_scratch: wgpu::Buffer,
    parameters: wgpu::Buffer,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    live: Vec<bool>,
    free_slots: Vec<u32>,
    last_submission: Option<wgpu::SubmissionIndex>,
}

impl WgpuPageBackend {
    fn new(max_pages: usize) -> Result<Self, WgpuBackendError> {
        let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
        descriptor.backends = wgpu::Backends::VULKAN
            | wgpu::Backends::METAL
            | wgpu::Backends::DX12
            | wgpu::Backends::GL;
        let instance = wgpu::Instance::new(descriptor);
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
            apply_limit_buckets: false,
        }))
        .map_err(|error| WgpuBackendError::Adapter(error.to_string()))?;
        let adapter_info = adapter.get_info();
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("seqcache wgpu example device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            ..Default::default()
        }))
        .map_err(|error| WgpuBackendError::Device(error.to_string()))?;

        let page_slab = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("seqcache physical page slab"),
            size: (max_pages * PAGE_TOKENS * size_of::<f32>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        // WebGPU copy validation does not permit a buffer to alias both sides
        // of one copy. The one-page scratch keeps tail copy-on-write portable
        // across native wgpu backends.
        let copy_scratch = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("seqcache partial-page copy scratch"),
            size: (PAGE_TOKENS * size_of::<f32>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let parameters = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("seqcache append parameters"),
            size: 4 * size_of::<u32>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: false,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("seqcache paged append shader"),
            source: wgpu::ShaderSource::Wgsl(WRITE_PAGED_ROWS_WGSL.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("seqcache paged append pipeline"),
            layout: None,
            module: &shader,
            entry_point: Some("write_paged_rows"),
            compilation_options: Default::default(),
            cache: None,
        });
        let bind_group_layout = pipeline.get_bind_group_layout(0);

        Ok(Self {
            device,
            queue,
            adapter_name: adapter_info.name,
            adapter_backend: adapter_info.backend,
            page_slab,
            copy_scratch,
            parameters,
            pipeline,
            bind_group_layout,
            live: vec![false; max_pages],
            free_slots: (0..max_pages as u32).rev().collect(),
            last_submission: None,
        })
    }

    fn new_page_table(&self, max_position: usize) -> WgpuPageTable {
        WgpuPageTable::new(&self.device, max_position)
    }

    fn is_live(&self, page: WgpuPage) -> bool {
        self.live.get(page.slot as usize).copied().unwrap_or(false)
    }

    fn validate_page(&self, page: WgpuPage) -> Result<(), WgpuBackendError> {
        if self.is_live(page) {
            Ok(())
        } else {
            Err(WgpuBackendError::StalePage(page.slot))
        }
    }

    fn validate_pages<'a>(
        &self,
        pages: impl IntoIterator<Item = &'a WgpuPage>,
    ) -> Result<(), WgpuBackendError> {
        for page in pages {
            self.validate_page(*page)?;
        }
        Ok(())
    }

    fn validate_distinct_pages(&self, pages: &[WgpuPage]) -> Result<(), WgpuBackendError> {
        self.validate_pages(pages)?;
        for (index, page) in pages.iter().enumerate() {
            if pages[..index].contains(page) {
                return Err(WgpuBackendError::InvalidGeometry(
                    "backend page batch contains a duplicate slot",
                ));
            }
        }
        Ok(())
    }

    fn allocate(&mut self) -> Result<PageAllocation<WgpuPage>, WgpuBackendError> {
        let slot = self.free_slots.pop().ok_or(WgpuBackendError::OutOfPages)?;
        self.live[slot as usize] = true;
        Ok(PageAllocation {
            page: WgpuPage { slot },
            recycled: true,
        })
    }

    fn release(&mut self, page: WgpuPage) -> Result<(), WgpuBackendError> {
        self.validate_page(page)?;
        self.live[page.slot as usize] = false;
        self.free_slots.push(page.slot);
        Ok(())
    }

    fn wait_for_gpu(&mut self) -> Result<(), WgpuBackendError> {
        let Some(submission) = self.last_submission.clone() else {
            return Ok(());
        };
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: None,
            })
            .map_err(|error| WgpuBackendError::Poll(error.to_string()))?;
        self.last_submission = None;
        Ok(())
    }

    fn publish_table(
        &mut self,
        pages: &[&WgpuPage],
        position: usize,
        context: &mut WgpuExecution<'_>,
    ) -> Result<(), WgpuBackendError> {
        let submission = context.page_table.publish(pages, position, &self.queue)?;
        self.last_submission = Some(submission);
        Ok(())
    }

    /// Stand-in for one model dispatch writing directly through the published table.
    fn dispatch_append(
        &mut self,
        context: &WgpuExecution<'_>,
        destinations: &[(WgpuPage, AppendSegment)],
        start_position: usize,
        rows: usize,
    ) -> Result<(), WgpuBackendError> {
        if rows == 0 || rows > u32::MAX as usize || start_position > u32::MAX as usize {
            return Err(WgpuBackendError::InvalidGeometry(
                "append dispatch exceeds the example's u32 geometry",
            ));
        }
        let covered_rows = destinations
            .iter()
            .try_fold(0usize, |total, (_, segment)| {
                total
                    .checked_add(segment.rows())
                    .ok_or(WgpuBackendError::InvalidGeometry(
                        "append row geometry overflowed",
                    ))
            })?;
        if covered_rows != rows {
            return Err(WgpuBackendError::InvalidGeometry(
                "append segments do not cover the dispatch",
            ));
        }
        for (page, segment) in destinations {
            self.validate_page(*page)?;
            let logical_page = (start_position + segment.input_offset()) / PAGE_TOKENS;
            if context.page_table.slots.get(logical_page).copied() != Some(page.slot) {
                return Err(WgpuBackendError::InvalidGeometry(
                    "published page table disagrees with append segments",
                ));
            }
        }

        let mut parameter_bytes = Vec::with_capacity(4 * size_of::<u32>());
        for value in [start_position as u32, rows as u32, PAGE_TOKENS as u32, 0] {
            parameter_bytes.extend_from_slice(&value.to_le_bytes());
        }
        self.queue
            .write_buffer(&self.parameters, 0, &parameter_bytes);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("seqcache paged append bindings"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.page_slab.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: context.page_table.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.parameters.as_entire_binding(),
                },
            ],
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("seqcache paged append commands"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("seqcache paged append pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(rows.div_ceil(WORKGROUP_SIZE) as u32, 1, 1);
        }
        self.last_submission = Some(self.queue.submit([encoder.finish()]));
        Ok(())
    }

    fn read_rows(
        &mut self,
        context: &WgpuExecution<'_>,
        rows: usize,
    ) -> Result<Vec<f32>, WgpuBackendError> {
        if rows > context.page_table.position {
            return Err(WgpuBackendError::InvalidGeometry(
                "cannot read beyond the committed position",
            ));
        }
        self.wait_for_gpu()?;
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("seqcache paged state readback"),
            size: (rows * size_of::<f32>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("seqcache paged state readback commands"),
            });
        for (logical_page, destination) in (0..rows).step_by(PAGE_TOKENS).enumerate() {
            let copied_rows = (rows - destination).min(PAGE_TOKENS);
            let physical_page = *context
                .page_table
                .slots
                .get(logical_page)
                .ok_or(WgpuBackendError::InvalidGeometry("page table is too short"))?;
            encoder.copy_buffer_to_buffer(
                &self.page_slab,
                physical_page as wgpu::BufferAddress
                    * PAGE_TOKENS as wgpu::BufferAddress
                    * size_of::<f32>() as wgpu::BufferAddress,
                &readback,
                destination as wgpu::BufferAddress * size_of::<f32>() as wgpu::BufferAddress,
                Some((copied_rows * size_of::<f32>()) as wgpu::BufferAddress),
            );
        }
        let submission = self.queue.submit([encoder.finish()]);
        let slice = readback.slice(..);
        let (sender, receiver) = mpsc::sync_channel(1);
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: None,
            })
            .map_err(|error| WgpuBackendError::Poll(error.to_string()))?;
        receiver
            .recv()
            .map_err(|error| WgpuBackendError::Map(error.to_string()))?
            .map_err(|error| WgpuBackendError::Map(error.to_string()))?;
        let mapped = slice
            .get_mapped_range()
            .map_err(|error| WgpuBackendError::Map(error.to_string()))?;
        let values = mapped
            .chunks_exact(size_of::<f32>())
            .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("one f32")))
            .collect();
        drop(mapped);
        readback.unmap();
        Ok(values)
    }
}

impl PageBackend for WgpuPageBackend {
    type Page = WgpuPage;
    type Context<'a> = WgpuExecution<'a>;
    // The shader writes only invalid rows. Abort and partial commit can hide
    // the unused suffix; a later append overwrites it before it becomes valid.
    type AppendTransaction = ();
    type Error = WgpuBackendError;

    fn page_bytes(&self) -> usize {
        PAGE_TOKENS * size_of::<f32>()
    }

    fn page_capacity(&self) -> Option<usize> {
        Some(self.live.len())
    }

    fn allocate_page(
        &mut self,
        _context: &mut Self::Context<'_>,
    ) -> Result<PageAllocation<Self::Page>, Self::Error> {
        self.allocate()
    }

    fn rollback_page(&mut self, page: Self::Page, _context: &mut Self::Context<'_>) {
        self.release(page)
            .expect("an unpublished wgpu page must remain live");
    }

    fn prepare_append(
        &mut self,
        pages: &[BackendAppendPage<'_, Self::Page>],
        _start_position: usize,
        _context: &mut Self::Context<'_>,
    ) -> Result<Self::AppendTransaction, Self::Error> {
        self.validate_pages(pages.iter().map(BackendAppendPage::page))?;
        if pages
            .iter()
            .any(|page| page.page_offset() + page.rows() > PAGE_TOKENS)
        {
            return Err(WgpuBackendError::InvalidGeometry(
                "append segment exceeds a physical wgpu page",
            ));
        }
        Ok(())
    }

    fn abort_append(
        &mut self,
        _transaction: &mut Self::AppendTransaction,
        restored_pages: &[&Self::Page],
        released_pages: &[&Self::Page],
        restored_position: usize,
        context: &mut Self::Context<'_>,
    ) -> Result<(), Self::Error> {
        self.validate_pages(restored_pages.iter().copied())?;
        self.validate_pages(released_pages.iter().copied())?;
        self.wait_for_gpu()?;
        self.publish_table(restored_pages, restored_position, context)?;
        for page in released_pages {
            self.release(**page)
                .expect("validated reservation page remains live");
        }
        Ok(())
    }

    fn copy_partial_page(
        &mut self,
        source: &Self::Page,
        valid_tokens: usize,
        _context: &mut Self::Context<'_>,
    ) -> Result<PageAllocation<Self::Page>, Self::Error> {
        self.validate_page(*source)?;
        if valid_tokens == 0 || valid_tokens >= PAGE_TOKENS {
            return Err(WgpuBackendError::InvalidGeometry(
                "partial-page copy requires an unaligned nonempty tail",
            ));
        }
        self.wait_for_gpu()?;
        let allocation = self.allocate()?;
        let bytes = (valid_tokens * size_of::<f32>()) as wgpu::BufferAddress;
        let source_offset = source.slot as wgpu::BufferAddress
            * PAGE_TOKENS as wgpu::BufferAddress
            * size_of::<f32>() as wgpu::BufferAddress;
        let destination_offset = allocation.page.slot as wgpu::BufferAddress
            * PAGE_TOKENS as wgpu::BufferAddress
            * size_of::<f32>() as wgpu::BufferAddress;
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("seqcache wgpu tail copy"),
            });
        encoder.copy_buffer_to_buffer(
            &self.page_slab,
            source_offset,
            &self.copy_scratch,
            0,
            Some(bytes),
        );
        encoder.copy_buffer_to_buffer(
            &self.copy_scratch,
            0,
            &self.page_slab,
            destination_offset,
            Some(bytes),
        );
        self.last_submission = Some(self.queue.submit([encoder.finish()]));
        if let Err(error) = self.wait_for_gpu() {
            self.release(allocation.page)
                .expect("failed copy destination remains live");
            return Err(error);
        }
        Ok(allocation)
    }

    fn commit_append(
        &mut self,
        _transaction: &mut Self::AppendTransaction,
        commit: BackendAppendCommit<'_, Self::Page>,
        context: &mut Self::Context<'_>,
    ) -> Result<(), Self::Error> {
        self.validate_pages(commit.committed_pages().iter().copied())?;
        self.validate_pages(commit.released_pages().iter().copied())?;
        self.wait_for_gpu()?;
        self.publish_table(commit.committed_pages(), commit.position(), context)?;
        for page in commit.released_pages() {
            self.release(**page)
                .expect("validated reservation page remains live");
        }
        Ok(())
    }

    fn update_page_table(
        &mut self,
        pages: &[&Self::Page],
        position: usize,
        context: &mut Self::Context<'_>,
    ) -> Result<(), Self::Error> {
        self.validate_pages(pages.iter().copied())?;
        self.publish_table(pages, position, context)
    }

    fn retire_pages(
        &mut self,
        pages: Vec<Self::Page>,
        _context: &mut Self::Context<'_>,
    ) -> Result<RetireOutcome, RetireError<Self::Error, Self::Page>> {
        if let Err(error) = self.validate_distinct_pages(&pages) {
            return Err(RetireError { error, pages });
        }
        if let Err(error) = self.wait_for_gpu() {
            return Err(RetireError { error, pages });
        }
        for page in pages {
            self.release(page)
                .expect("validated retirement page remains live");
        }
        Ok(RetireOutcome::default())
    }

    fn retirement_is_immediate(&self) -> bool {
        true
    }

    fn poll_reclaimed(&mut self, _context: &mut Self::Context<'_>) -> Result<usize, Self::Error> {
        Ok(0)
    }
}

type WgpuSequenceCache = SequenceCache<WgpuPageBackend, ()>;

fn admitted(outcome: AdmissionOutcome) -> Result<SequenceId, Box<dyn Error>> {
    match outcome {
        AdmissionOutcome::Admitted(sequence) => Ok(sequence),
        AdmissionOutcome::WouldBlock => Err("cache admission unexpectedly blocked".into()),
    }
}

fn request(page_table: &WgpuPageTable) -> AdmissionRequest {
    AdmissionRequest {
        max_position: MAX_SEQUENCE_TOKENS,
        private_state_bytes: 0,
        page_table_bytes: page_table.managed_bytes(),
        allow_emergency: false,
    }
}

fn append(
    cache: &mut WgpuSequenceCache,
    sequence: SequenceId,
    rows: usize,
    context: &mut WgpuExecution<'_>,
) -> Result<usize, Box<dyn Error>> {
    let reservation = cache.reserve_append(sequence, rows, context)?;
    let start_position = reservation.start_position();
    let segment_count = reservation.segments().len();
    cache.with_append_pages(&reservation, |backend, pages| {
        let destinations = pages
            .iter()
            .map(|page| (*page.page(), page.segment()))
            .collect::<Vec<_>>();
        backend.dispatch_append(context, &destinations, start_position, rows)
    })?;
    cache.commit_append(reservation, rows, context)?;
    Ok(segment_count)
}

fn main() -> Result<(), Box<dyn Error>> {
    let backend = WgpuPageBackend::new(MAX_PHYSICAL_PAGES)?;
    let adapter_name = backend.adapter_name.clone();
    let adapter_backend = backend.adapter_backend;
    let mut source_table = backend.new_page_table(MAX_SEQUENCE_TOKENS);
    let page_bytes = backend.page_bytes();
    let config = CacheConfig {
        page_tokens: PAGE_TOKENS,
        max_managed_bytes: page_bytes * MAX_PHYSICAL_PAGES + 4_096,
        max_snapshot_bytes: 0,
        max_prefix_entries: None,
        emergency_bytes: 0,
    };
    let mut cache = SequenceCache::new(config, backend)?;
    let source_request = request(&source_table);
    let mut source_context = WgpuExecution {
        page_table: &mut source_table,
    };
    let source = admitted(cache.admit(
        None,
        source_request,
        &mut source_context,
        |snapshot, position| {
            assert!(snapshot.is_none());
            assert_eq!(position, 0);
            Ok(())
        },
    )?)?;

    // Leave a partial tail, then cross two page boundaries in one dispatch.
    assert_eq!(append(&mut cache, source, 2, &mut source_context)?, 1);
    let segments = append(&mut cache, source, 9, &mut source_context)?;
    assert_eq!(segments, 3);
    assert_eq!(source_context.page_table.position, 11);
    let source_rows = cache.with_backend(|backend| backend.read_rows(&source_context, 11))?;
    assert_eq!(
        source_rows,
        (1..=11).map(|value| value as f32).collect::<Vec<_>>()
    );

    // Exercise the backend's GPU tail-copy path. Complete pages remain shared;
    // only the three-row unaligned tail is copied for the branch.
    let mut branch_table =
        cache.with_backend(|backend| backend.new_page_table(MAX_SEQUENCE_TOKENS));
    let branch_request = request(&branch_table);
    let mut branch_context = WgpuExecution {
        page_table: &mut branch_table,
    };
    let branch = admitted(cache.branch(source, branch_request, &mut branch_context)?)?;
    assert_eq!(append(&mut cache, branch, 1, &mut branch_context)?, 1);
    let branch_rows = cache.with_backend(|backend| backend.read_rows(&branch_context, 12))?;
    assert_eq!(
        branch_rows,
        (1..=12).map(|value| value as f32).collect::<Vec<_>>()
    );
    assert_eq!(cache.page_table(source)?.position(), 11);
    cache.validate()?;

    println!("adapter: {adapter_name} ({adapter_backend:?})");
    println!(
        "wrote 9 rows through {segments} wgpu pages in one dispatch; source position: {}",
        source_context.page_table.position,
    );
    println!(
        "branched the unaligned tail with one GPU copy; branch position: {}",
        branch_context.page_table.position,
    );

    cache.finish(branch, &mut branch_context)?;
    cache.finish(source, &mut source_context)?;
    cache.validate()?;
    Ok(())
}
