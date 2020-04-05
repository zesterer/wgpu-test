mod render;

use std::time::{Instant, Duration};
use vek::*;
use winit::{
    event::*,
    event_loop::{EventLoop, ControlFlow},
    window::WindowBuilder,
};
use crate::render::*;

fn main() {
    env_logger::init();

    let event_loop = EventLoop::new();
    let win = WindowBuilder::new()
        .build(&event_loop)
        .unwrap();

    let mut renderer = async_std::task::block_on(Renderer::new(&win));

    #[repr(C)]
    #[derive(Copy, Clone, Default, Debug)]
    struct Vert {
        pos: [f32; 3],
        tex: [f32; 2],
    }

    impl Vertex for Vert {
        fn attr_desc(&self, index: usize) -> Option<(wgpu::VertexFormat, usize)> {
            [self.pos.desc(), self.tex.desc()].get(index).copied()
        }
    }

    let mesh = [
        Vert { pos: [ 0.0, -0.5, 0.0], tex: [ 0.5, 0.0] },
        Vert { pos: [-0.5,  0.5, 0.0], tex: [-1.0, 1.0] },
        Vert { pos: [ 0.5,  0.5, 0.0], tex: [ 1.0, 1.0] },
    ].iter().copied().collect();

    #[derive(Copy, Clone)]
    struct Globals {
        view_proj: [f32; 16],
    }

    struct TriPipeline;

    impl Pipeline for TriPipeline {
        type Vertex = Vert;
        type BindingLayout = (Uniforms<Globals>,);
    }

    let vs = renderer.create_shader(include_str!("tri.vert")).unwrap();
    let fs = renderer.create_shader(include_str!("tri.frag")).unwrap();

    let pipeline = renderer.create_pipeline::<TriPipeline>(
        &vs,
        &fs,
    );

    let uniforms = renderer.create_uniforms(Globals {
        view_proj: Mat4::identity().into_col_array(),
    });

    let bindings = renderer.create_bindings(&pipeline, &(uniforms,));

    let model = renderer.create_model(&mesh);

    event_loop.run(move |event, _, cf| {
        match event {
            Event::WindowEvent { ref event, window_id } if window_id == win.id() => match event {
                WindowEvent::CloseRequested => *cf = ControlFlow::Exit,
                WindowEvent::Resized(physical_size) => renderer.resize(*physical_size),
                WindowEvent::ScaleFactorChanged { new_inner_size, .. } => renderer.resize(**new_inner_size),
                _ => {},
            },
            Event::NewEvents(_) => *cf = ControlFlow::Wait,
            Event::MainEventsCleared => {
                renderer.clear(Rgba::white());
                renderer.render(RenderTask {
                    model: &model,
                    pipeline: &pipeline,
                    bindings: &bindings,
                });
                *cf = ControlFlow::WaitUntil(Instant::now() + Duration::from_millis(1000 / 60))
            },
            _ => {},
        }
    });
}
