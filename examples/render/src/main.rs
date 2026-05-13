//! Example of using [`naga_rust_embed`] to execute a fragment shader on the CPU.
//! This is not very efficient, but it works.

#![allow(clippy::pedantic)]

use std::sync::Arc;
use std::time::Instant;

use rayon::iter::IndexedParallelIterator;
use rayon::{iter::ParallelIterator as _, slice::ParallelSliceMut as _};
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

use naga_rust_embed::rt;

naga_rust_embed::include_wgsl_mr!(resource_struct = Shader, "src/shader.wgsl");

// -------------------------------------------------------------------------------------------------

fn main() {
    let event_loop = EventLoop::new().unwrap();
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_with_display_handle_from_env(
        Box::new(event_loop.owned_display_handle()),
    ));
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop
        .run_app(&mut App {
            instance,
            active: None,
            start_time: Instant::now(),
        })
        .unwrap();
}

// -------------------------------------------------------------------------------------------------

fn run_fragment_shader(time: f32, buffer: &mut [[u8; 4]], size: PhysicalSize<u32>) {
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
                #[allow(clippy::cast_precision_loss)]
                let fragment_position = rt::Vec4::new(x as f32, y as f32, 0.0, 0.0);

                let result = Shader {
                    time: rt::Scalar(time),
                }
                .main(fragment_position);

                // In a real application this should be sRGB encoding.
                let v = (result * rt::Scalar(255.0)).map(|c: f32| c as u8);

                *pixel = [v.z, v.y, v.x, 0xFF];
            }
        });
}

// -------------------------------------------------------------------------------------------------

struct App {
    instance: wgpu::Instance,
    active: Option<Active>,
    start_time: Instant,
}
struct Active {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    cpu_buffer: Vec<[u8; 4]>,
    device: wgpu::Device,
    queue: wgpu::Queue,
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
            let surface = self.instance.create_surface(window.clone()).unwrap();
            let adapter = pollster::block_on(wgpu::util::initialize_adapter_from_env_or_default(
                &self.instance,
                Some(&surface),
            ))
            .unwrap();
            let (device, queue) =
                pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                    required_limits: wgpu::Limits::default().using_resolution(adapter.limits()),
                    ..Default::default()
                }))
                .unwrap();

            let active = Active {
                window,
                surface,
                device,
                queue,
                cpu_buffer: Vec::new(),
            };
            active.configure_surface();

            self.active = Some(active);
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
            WindowEvent::Resized(_) => {
                if let Some(active) = &self.active {
                    active.configure_surface();
                }
            }
            WindowEvent::RedrawRequested => {
                if let Some(active) = &mut self.active {
                    let time = self.start_time.elapsed().as_secs_f32();
                    match active.surface.get_current_texture() {
                        wgpu::CurrentSurfaceTexture::Success(surface_texture)
                        | wgpu::CurrentSurfaceTexture::Suboptimal(surface_texture) => {
                            active.render(time, surface_texture)
                        }
                        wgpu::CurrentSurfaceTexture::Occluded => {}
                        //t => panic!("{t:?}"),
                        wgpu::CurrentSurfaceTexture::Timeout => {}
                        wgpu::CurrentSurfaceTexture::Outdated => {
                            active.configure_surface();
                        }
                        wgpu::CurrentSurfaceTexture::Lost => {
                            todo!("recreating device not implemented")
                        }
                        wgpu::CurrentSurfaceTexture::Validation => {
                            unreachable!("we are not catching errors")
                        }
                    }
                }

                // Continuous animation.
                if let Some(Active { window, .. }) = &self.active {
                    window.request_redraw();
                }
            }
            _ => (),
        }
    }
}

impl Active {
    fn configure_surface(&self) {
        let size = self.window.inner_size();
        self.surface.configure(
            &self.device,
            &wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::COPY_DST,
                // guaranteed to work per https://docs.rs/wgpu/29.0.3/wgpu/type.SurfaceConfiguration.html#structfield.format
                format: wgpu::TextureFormat::Bgra8Unorm,
                width: size.width,
                height: size.height,
                present_mode: wgpu::PresentMode::AutoVsync,
                desired_maximum_frame_latency: 3,
                alpha_mode: wgpu::CompositeAlphaMode::Opaque,
                view_formats: vec![wgpu::TextureFormat::Bgra8UnormSrgb],
            },
        );
    }

    fn render(&mut self, time: f32, surface_texture: wgpu::SurfaceTexture) {
        let size = PhysicalSize {
            width: surface_texture.texture.width(),
            height: surface_texture.texture.height(),
        };

        // Run shader (on CPU)
        self.cpu_buffer
            .resize(size.width as usize * size.height as usize, [0; 4]);
        run_fragment_shader(time, &mut self.cpu_buffer, size);

        // Copy data from CPU memory to GPU memory.
        //
        // TODO: It would be more efficient to use write_buffer_with() and a texture copy command.
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &surface_texture.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            self.cpu_buffer.as_flattened(),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(size.width * 4),
                rows_per_image: None,
            },
            wgpu::Extent3d {
                width: size.width,
                height: size.height,
                depth_or_array_layers: 1,
            },
        );

        // write_texture() is deferred to the next submit. Usually, this doesn’t matter because
        // there is at least one command submitted, such as a render pass, that uses the written
        // data, but in this case we are writing directly to the surface texture, so we need a
        // dummy submit to flush the write.
        self.queue.submit([]);

        surface_texture.present();
    }
}
