use winit::event_loop::{EventLoop, ControlFlow};
use winit::window::WindowBuilder;
use winit::event::{Event, WindowEvent};
use std::sync::Arc;
use nexir::render::device::GpuDevice;
use log::info;

mod app;
mod history;
pub mod layout;

fn main() {
    env_logger::init();
    info!("Starting Nexir UI");

    let event_loop = EventLoop::new().unwrap();
    let window = Arc::new(WindowBuilder::new()
        .with_title("Nexir Video Editor")
        .with_inner_size(winit::dpi::LogicalSize::new(1280.0, 720.0))
        .build(&event_loop)
        .unwrap());

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

    event_loop.run(move |event, elwt| {
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
                            device.configure_surface(&surface, physical_size.width, physical_size.height);
                        }
                    }
                    WindowEvent::RedrawRequested => {
                        let viewport_size = app_state.update(&window);
                        app_state.render(&*device, &surface, &window, viewport_size);
                    }
                    _ => {}
                }
            }
            Event::AboutToWait => {
                window.request_redraw();
            }
            _ => {}
        }
    }).unwrap();
}
