//! Example of using [`naga_rust_embed`] to execute a fragment shader on the CPU.
//! This is not very efficient, but it works.

use std::num::NonZeroU32;
use std::sync::Arc;
use std::time::Instant;

use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

naga_rust_embed::include_wgsl_mr!(global_struct = Shader, "src/shader.wgsl");

// -------------------------------------------------------------------------------------------------

fn main() {
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop
        .run_app(&mut App {
            active: None,
            start_time: Instant::now(),
        })
        .unwrap();
}

// -------------------------------------------------------------------------------------------------

struct App {
    active: Option<Active>,
    start_time: Instant,
}
struct Active {
    window: Arc<Window>,
    surface: softbuffer::Surface<Arc<Window>, Arc<Window>>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.active.is_none() {
            let window = Arc::new(
                event_loop
                    .create_window(
                        Window::default_attributes()
                            .with_title("CPU shader execution example")
                            .with_inner_size(winit::dpi::LogicalSize::new(300, 300)),
                    )
                    .unwrap(),
            );
            let context = softbuffer::Context::new(window.clone()).unwrap();
            let surface = softbuffer::Surface::new(&context, window.clone()).unwrap();

            self.active = Some(Active { window, surface });
        }
    }

    fn suspended(&mut self, _: &ActiveEventLoop) {
        self.active = None;
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                if let Some(Active { window, surface }) = &mut self.active {
                    let t = self.start_time.elapsed().as_secs_f32();

                    let (width, height) = {
                        let size = window.inner_size();
                        (size.width, size.height)
                    };
                    surface
                        .resize(
                            NonZeroU32::new(width).unwrap(),
                            NonZeroU32::new(height).unwrap(),
                        )
                        .unwrap();

                    let mut buffer = surface.buffer_mut().unwrap();
                    for index in 0..(width * height) {
                        let y = index / width;
                        let x = index % width;
                        // TODO: `rt` is not supposed to be in scope here
                        #[allow(clippy::cast_precision_loss)]
                        let fragment_position = rt::Vec4::new(x as f32, y as f32, 0.0, 0.0);

                        let shader = Shader::default();
                        let result = shader.main(t, fragment_position);

                        // In a real application this should be sRGB encoding.
                        let v = (result * 255.0).as_uvec4().map(|c| c.clamp(0, 255));

                        buffer[index as usize] = v.z | (v.y << 8) | (v.x << 16);
                    }

                    buffer.present().unwrap();

                    // Continuous animation.
                    window.request_redraw();
                }
            }
            _ => (),
        }
    }
}
