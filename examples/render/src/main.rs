//! Example of using [`naga_rust_embed`] to execute a fragment shader on the CPU.
//! This is not very efficient, but it works.

use std::num::NonZeroU32;
use std::sync::Arc;
use std::time::Instant;

use rayon::iter::IndexedParallelIterator;
use rayon::{iter::ParallelIterator as _, slice::ParallelSliceMut as _};
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
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

fn run_fragment_shader(time: f32, buffer: &mut [u32], size: PhysicalSize<u32>) {
    // Run in parallel to take advantage of all CPU cores.
    //
    // We parallelize rows but run each row serially to ensure that the inner loop has no
    // overhead from parallelization and can be unrolled or whatever the compiler wants to
    // do with it, because *each pixel* is trivial but there are a lot of them, so the most
    // important thing is minimizing overhead.
    //
    // The other big thing for efficiency is to arrange for multiple pixels to be computed
    // in SIMD, but we are not yet doing that, or rather, the compiler may manage to
    // autovectorize the code but we’re not doing anything to ensure it has an easy job of that.

    buffer
        .par_chunks_mut(size.width as usize)
        .enumerate()
        .for_each(|(y, row)| {
            for (x, pixel) in row.iter_mut().enumerate() {
                // TODO: `rt` is not supposed to be in scope here
                #[allow(clippy::cast_precision_loss)]
                let fragment_position = rt::Vec4::new(x as f32, y as f32, 0.0, 0.0);

                let result = Shader { time }.main(fragment_position);

                // In a real application this should be sRGB encoding.
                let v = (result * 255.0).cast_elem_as_u32().map(|c| c.clamp(0, 255));

                *pixel = v.z | (v.y << 8) | (v.x << 16);
            }
        });
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
                    let time = self.start_time.elapsed().as_secs_f32();
                    let size = window.inner_size();
                    surface
                        .resize(
                            NonZeroU32::new(size.width).unwrap(),
                            NonZeroU32::new(size.height).unwrap(),
                        )
                        .unwrap();

                    let mut buffer = surface.buffer_mut().unwrap();
                    run_fragment_shader(time, &mut buffer, size);

                    buffer.present().unwrap();

                    // Continuous animation.
                    window.request_redraw();
                }
            }
            _ => (),
        }
    }
}
