use log::info;
use nexir::render::device::GpuDevice;
use std::sync::Arc;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowBuilder;

mod app;
mod audio_mixer;
mod autosave;
mod history;
mod image_still;
pub mod layout;
mod waveform;

struct SimpleFileLogger {
    file: std::sync::Mutex<Option<std::fs::File>>,
}

impl log::Log for SimpleFileLogger {
    fn enabled(&self, metadata: &log::Metadata) -> bool {
        // Filter out spam from wgpu/naga
        let target = metadata.target();
        if target.starts_with("wgpu") || target.starts_with("naga") {
            return metadata.level() <= log::Level::Error;
        }
        metadata.level() <= log::Level::Info
    }

    fn log(&self, record: &log::Record) {
        if !self.enabled(record.metadata()) {
            return;
        }

        let msg = format!(
            "[{}] {} - {}\n",
            record.level(),
            record.target(),
            record.args()
        );

        // Print to stderr
        eprint!("{}", msg);

        // Write to file
        if let Ok(mut guard) = self.file.lock()
            && let Some(file) = guard.as_mut() {
                use std::io::Write;
                let _ = file.write_all(msg.as_bytes());
                let _ = file.flush();
            }
    }

    fn flush(&self) {}
}

fn main() {
    // Setup crash logging and standard logging to a file
    let log_file = if let Ok(exe_path) = std::env::current_exe() {
        if let Some(dir) = exe_path.parent() {
            let log_path = dir.join("nexir_log.txt");
            std::fs::OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(log_path)
                .ok()
        } else {
            None
        }
    } else {
        None
    };

    // Initialize the logger
    let logger = SimpleFileLogger {
        file: std::sync::Mutex::new(log_file),
    };
    log::set_boxed_logger(Box::new(logger)).unwrap();
    log::set_max_level(log::LevelFilter::Info);

    // Setup custom panic hook
    std::panic::set_hook(Box::new(|panic_info| {
        let mut message = String::new();
        if let Some(s) = panic_info.payload().downcast_ref::<&str>() {
            message.push_str(s);
        } else if let Some(s) = panic_info.payload().downcast_ref::<String>() {
            message.push_str(s);
        } else {
            message.push_str("Unknown panic");
        }

        let location = if let Some(loc) = panic_info.location() {
            format!("at {}:{}", loc.file(), loc.line())
        } else {
            "unknown location".to_string()
        };

        let backtrace = std::backtrace::Backtrace::capture();
        let log_content = format!(
            "==================================================\n\
             NEXIR CRASH REPORT\n\
             ==================================================\n\
             Panic: {}\n\
             Location: {}\n\
             Backtrace:\n\
             {:#?}\n",
            message, location, backtrace
        );

        eprint!("{}", log_content);

        if let Ok(exe_path) = std::env::current_exe()
            && let Some(dir) = exe_path.parent() {
                let log_path = dir.join("nexir_crash.log");
                let _ = std::fs::write(log_path, log_content);
            }
    }));

    info!("Starting Nexir UI");

    let event_loop = EventLoop::new().unwrap();
    let window = Arc::new(
        WindowBuilder::new()
            .with_title("Nexir Video Editor")
            .with_inner_size(winit::dpi::LogicalSize::new(1280.0, 720.0))
            .build(&event_loop)
            .unwrap(),
    );

    let size = window.inner_size();

    // Initialize the engine and GPU device
    let (device, surface) = pollster::block_on(async {
        GpuDevice::new_with_surface(window.clone().into(), size.width, size.height)
            .await
            .expect("Failed to initialize GPU device")
    });

    device.configure_surface(&surface, size.width, size.height);
    let device = Arc::new(device);

    let mut app_state = app::NexirApp::new(Arc::clone(&device), &window);

    event_loop
        .run(move |event, elwt| {
            elwt.set_control_flow(ControlFlow::Poll);

            match event {
                Event::WindowEvent { event, window_id } if window_id == window.id() => {
                    let response = app_state.handle_event(&window, &event);
                    if response.consumed {
                        return;
                    }

                    match event {
                        WindowEvent::CloseRequested => elwt.exit(),
                        WindowEvent::Resized(physical_size) => {
                            if physical_size.width > 0 && physical_size.height > 0 {
                                device.configure_surface(
                                    &surface,
                                    physical_size.width,
                                    physical_size.height,
                                );
                            }
                        }
                        WindowEvent::RedrawRequested => {
                            let viewport_size = app_state.update(&window);
                            app_state.render(&device, &surface, &window, viewport_size);
                        }
                        _ => {}
                    }
                }
                Event::AboutToWait => {
                    window.request_redraw();
                }
                _ => {}
            }
        })
        .unwrap();
}
