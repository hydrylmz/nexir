// src/bin/bench.rs
// Nexir render pipeline benchmark
// Run with: cargo run -p nexir --bin bench --release

use std::sync::Arc;
use std::time::{Duration, Instant};
use nexir::render::device::GpuDevice;
use nexir::render::graph::RenderGraphCompiler;
use nexir::render::nodes::composite::CompositeNode;
use nexir::render::nodes::yuv_to_rgb::YuvToRgbNode;
use nexir::render::nodes::yuv_upload::YuvUploadNode;
use nexir::render::resource::ResourceId;
use nexir::render::shader::registry::ShaderRegistry;
use nexir::render::compute::ComputePipelineCache;
use nexir::render::frame_state::{FrameState, ClipRenderEntry};
use nexir::timeline::ids::SourceId;
use nexir::timeline::transform::ClipTransform;
use nexir::timeline::source::{ColorInfo, ColorRange, MatrixCoefficients, TransferFunction, ColorPrimaries};
use nexir::timeline::store::ClipKind;

struct Scenario {
    name: &'static str,
    canvas_w: u32, canvas_h: u32,
    clip_w: u32,   clip_h: u32,
    frames: usize,
    with_readback: bool,
}

const SCENARIOS: &[Scenario] = &[
    Scenario { name: "1080p  NVENC path", canvas_w: 1920, canvas_h: 1080, clip_w: 1920, clip_h: 1080, frames: 300, with_readback: false },
    Scenario { name: "4K     NVENC path", canvas_w: 3840, canvas_h: 2160, clip_w: 3840, clip_h: 2160, frames: 180, with_readback: false },
    Scenario { name: "1080p  CPU   path", canvas_w: 1920, canvas_h: 1080, clip_w: 1920, clip_h: 1080, frames: 120, with_readback: true  },
];

fn make_nv12(w: u32, h: u32) -> Vec<u8> {
    let y = (w * h) as usize;
    let mut b = vec![200u8; y + y / 2];
    for x in &mut b[y..] { *x = 128; }
    b
}
fn align256(n: u32) -> u32 { (n + 255) & !255 }
fn stats(t: &[Duration]) -> (f64, f64, f64) {
    let total: f64 = t.iter().map(|d| d.as_secs_f64()).sum();
    let fps = t.len() as f64 / total;
    let mean = total / t.len() as f64 * 1000.0;
    let mut s = t.to_vec(); s.sort_unstable();
    let p99 = s[(s.len()*99/100).min(s.len()-1)].as_secs_f64()*1000.0;
    (fps, mean, p99)
}

fn run(device: &Arc<GpuDevice>, s: &Scenario) -> Vec<Duration> {
    let shaders = ShaderRegistry::compile_all(device).expect("shader compile");
    let compute = Arc::new(ComputePipelineCache::new());
    let ci = ColorInfo { matrix: MatrixCoefficients::Bt709, range: ColorRange::Limited,
        transfer_fn: TransferFunction::Bt709, primaries: ColorPrimaries::Bt709, bit_depth: 8 };
    let mut compiler = RenderGraphCompiler::new();
    let mut idc = 2u32;
    let rgba_id = ResourceId::next(&mut idc);
    let y_id = ResourceId::next(&mut idc);
    let uv_id = ResourceId::next(&mut idc);
    let uidx = compiler.add_node(Box::new(YuvUploadNode::new(device, 0, s.clip_w, s.clip_h, y_id, uv_id)));
    compiler.add_node(Box::new(YuvToRgbNode::new(device, &shaders, &compute, y_id, uv_id, rgba_id, s.clip_w, s.clip_h, ci)));
    let mut cn = CompositeNode::new(device, &shaders, ResourceId::FINAL_COLOR, 1, wgpu::TextureFormat::Rgba16Float);
    cn.input_textures.push(rgba_id);
    compiler.add_node(Box::new(cn));
    let mut graph = compiler.compile(s.canvas_w, s.canvas_h).expect("compile");
    let rb: Option<wgpu::Buffer> = s.with_readback.then(|| {
        let stride = align256(s.canvas_w * 8);
        device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rb"), size: stride as u64 * s.canvas_h as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        })
    });
    let nv12 = make_nv12(s.clip_w, s.clip_h);
    let fs = FrameState {
        pts: 0, canvas_width: s.canvas_w, canvas_height: s.canvas_h,
        clips: vec![ClipRenderEntry { source_id: SourceId::new(0), texture_slot: 0, layer_order: 0,
            clip_width: s.clip_w, clip_height: s.clip_h, transform: ClipTransform::identity(),
            opacity: 1.0, is_nv12: true, kind: ClipKind::Video }],
        test_textures: vec![],
    };
    for _ in 0..3 {
        if let Some(u) = graph.nodes_mut()[uidx].as_any_mut().and_then(|n| n.downcast_mut::<YuvUploadNode>()) {
            u.upload_frame(&nv12, true, s.clip_w, s.clip_h);
        }
        let mut e = device.begin_frame(); graph.execute(&mut e, device, &fs);
        let sid = device.submit(e); device.device.poll(wgpu::Maintain::WaitForSubmissionIndex(sid));
    }
    let mut times = Vec::with_capacity(s.frames);
    for _ in 0..s.frames {
        let t0 = Instant::now();
        if let Some(u) = graph.nodes_mut()[uidx].as_any_mut().and_then(|n| n.downcast_mut::<YuvUploadNode>()) {
            u.upload_frame(&nv12, true, s.clip_w, s.clip_h);
        }
        let mut e = device.begin_frame();
        if let Some(buf) = &rb {
            graph.execute_with_callback(&mut e, device, &fs, |enc, ctx| {
                if ctx.contains(ResourceId::FINAL_COLOR) {
                    let stride = align256(s.canvas_w * 8);
                    enc.copy_texture_to_buffer(
                        wgpu::ImageCopyTexture { texture: ctx.get(ResourceId::FINAL_COLOR).texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                        wgpu::ImageCopyBuffer { buffer: buf, layout: wgpu::ImageDataLayout { offset: 0, bytes_per_row: Some(stride), rows_per_image: Some(s.canvas_h) } },
                        wgpu::Extent3d { width: s.canvas_w, height: s.canvas_h, depth_or_array_layers: 1 },
                    );
                }
            });
        } else { graph.execute(&mut e, device, &fs); }
        let sid = device.submit(e);
        if s.with_readback {
            device.device.poll(wgpu::Maintain::WaitForSubmissionIndex(sid));
            if let Some(buf) = &rb {
                let sl = buf.slice(..);
                let (tx, rx) = std::sync::mpsc::channel();
                sl.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
                device.device.poll(wgpu::Maintain::Wait);
                rx.recv().unwrap().unwrap();
                let v = sl.get_mapped_range();
                let _: u64 = v.chunks(8).take(16).map(|c| c[0] as u64).sum();
                drop(v); buf.unmap();
            }
        } else { device.device.poll(wgpu::Maintain::WaitForSubmissionIndex(sid)); }
        times.push(t0.elapsed());
    }
    times
}

fn main() {
    println!("\n  Nexir Render Pipeline Benchmark\n  ================================\n");
    let device = Arc::new(pollster::block_on(GpuDevice::new_headless()).expect("no GPU"));
    let info = device.adapter.get_info();
    println!("  GPU     : {} ({:?})", info.name, info.backend);
    println!("  BindArr : {}\n", device.has_binding_arrays);
    println!("  {:<40}  {:>9}  {:>10}  {:>10}", "Scenario", "FPS", "Mean ms", "p99 ms");
    println!("  {}", "-".repeat(76));
    for s in SCENARIOS {
        let (fps, mean, p99) = stats(&run(&device, s));
        println!("  {:<40}  {:>9.1}  {:>9.2}  {:>9.2}", s.name, fps, mean, p99);
    }
    println!("  {}\n", "-".repeat(76));
}
