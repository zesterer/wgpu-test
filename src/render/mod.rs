use std::{
    marker::PhantomData,
    iter::FromIterator,
};
use vek::*;
use winit::{
    window::Window,
    dpi::PhysicalSize,
};

#[derive(Debug)]
pub enum RenderError {
    ShaderCompilationError(String),
}

pub struct Renderer {
    surface: wgpu::Surface,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,

    swap_chain_desc: wgpu::SwapChainDescriptor,
    swap_chain: wgpu::SwapChain,
}

impl Renderer {
    pub async fn new(win: &Window) -> Self {
        let surface = wgpu::Surface::create(win);
        let adapter = wgpu::Adapter::request(
            &wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
            },
            wgpu::BackendBit::all(),
        )
            .await
            .unwrap();
        let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor {
            extensions: wgpu::Extensions {
                anisotropic_filtering: false,
            },
            limits: wgpu::Limits::default(),
        })
            .await;
        let swap_chain_desc = wgpu::SwapChainDescriptor {
            usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
            format: wgpu::TextureFormat::Bgra8UnormSrgb,
            width: win.inner_size().width,
            height: win.inner_size().height,
            present_mode: wgpu::PresentMode::Fifo,
        };
        let swap_chain = device.create_swap_chain(&surface, &swap_chain_desc);

        Self {
            surface,
            adapter,
            device,
            queue,

            swap_chain_desc,
            swap_chain,
        }
    }

    pub fn resize(&mut self, new_size: PhysicalSize<u32>) {
        self.swap_chain_desc.width = new_size.width;
        self.swap_chain_desc.height = new_size.height;
        self.swap_chain = self.device.create_swap_chain(&self.surface, &self.swap_chain_desc);
    }

    pub fn create_shader<P: Pipeline, S: Stage>(&self, code: &str) -> Result<Shader<P, S>, RenderError> {
        let spirv = glsl_to_spirv::compile(code, S::STAGE)
            .map_err(RenderError::ShaderCompilationError)?;
        let data = wgpu::read_spirv(spirv).expect("Shader compiled emitted invalid SPIR-V");
        Ok(Shader {
            module: self.device.create_shader_module(&data),
            _phantom: PhantomData,
        })
    }

    pub fn create_pipeline<P: Pipeline>(
        &self,
        vs: &Shader<P, Vert>,
        fs: &Shader<P, Frag>,
    ) -> PipelineState<P> {
        let binding_layout = P::BindingLayout::do_for_layout(|layout| {
            self.device.create_bind_group_layout(&layout)
        });

        let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            bind_group_layouts: &[&binding_layout],
        });

        println!("HERE0!");

        let pipeline = P::Vertex::default().desc(|vertex_desc| {
            self.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                layout: &pipeline_layout,
                vertex_stage: wgpu::ProgrammableStageDescriptor {
                    module: &vs.module,
                    entry_point: "main",
                },
                fragment_stage: Some(wgpu::ProgrammableStageDescriptor {
                    module: &fs.module,
                    entry_point: "main",
                }),
                rasterization_state: Some(wgpu::RasterizationStateDescriptor {
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: wgpu::CullMode::Back,
                    depth_bias: 0,
                    depth_bias_slope_scale: 0.0,
                    depth_bias_clamp: 0.0,
                }),
                color_states: &[
                    wgpu::ColorStateDescriptor {
                        format: self.swap_chain_desc.format,
                        color_blend: wgpu::BlendDescriptor::REPLACE,
                        alpha_blend: wgpu::BlendDescriptor::REPLACE,
                        write_mask: wgpu::ColorWrite::ALL,
                    },
                ],
                primitive_topology: wgpu::PrimitiveTopology::TriangleList,
                depth_stencil_state: None,
                vertex_state: wgpu::VertexStateDescriptor {
                    index_format: wgpu::IndexFormat::Uint32,
                    vertex_buffers: &[vertex_desc],
                },
                sample_count: 1,
                sample_mask: !0,
                alpha_to_coverage_enabled: false,
            })
        });

        println!("HERE1!");

        PipelineState {
            pipeline,
            binding_layout,
            _phantom: PhantomData,
        }
    }

    pub fn create_model<V: Vertex>(&self, mesh: &Mesh<V>) -> Model<V> {
        Model {
            buf: self.device
                .create_buffer_with_data(unsafe { std::slice::from_raw_parts(mesh.verts.as_ptr() as *const _, mesh.verts.len() * std::mem::size_of::<V>()) }, wgpu::BufferUsage::VERTEX),
            vert_count: mesh.verts.len(),
            _phantom: PhantomData,
        }
    }

    pub fn create_uniforms<U: Copy + 'static>(&self, uniforms: U) -> Uniforms<U> {
        Uniforms {
            buf: self.device
                .create_buffer_with_data(unsafe { std::slice::from_raw_parts(&uniforms as *const _ as *const _, std::mem::size_of::<U>()) }, wgpu::BufferUsage::UNIFORM),
            _phantom: PhantomData,
        }
    }

    pub fn create_bindings<P: Pipeline>(
        &self,
        pipeline_state: &PipelineState<P>,
        layout: &P::BindingLayout,
    ) -> Bindings<P::BindingLayout> {
        Bindings {
            group: layout.do_for_group(&pipeline_state.binding_layout, |group| self.device.create_bind_group(&group)),
            _phantom: PhantomData,
        }
    }

    pub fn clear(&mut self, col: Rgba<f64>) {
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: None,
        });
        let frame = self.swap_chain.get_next_texture().unwrap();
        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            color_attachments: &[
                wgpu::RenderPassColorAttachmentDescriptor {
                    attachment: &frame.view,
                    resolve_target: None,
                    load_op: wgpu::LoadOp::Clear,
                    store_op: wgpu::StoreOp::Store,
                    clear_color: wgpu::Color { r: col.r, g: col.g, b: col.g, a: col.a },
                },
            ],
            depth_stencil_attachment: None,
        });

        self.queue.submit(&[encoder.finish()]);
    }

    pub fn render_batch<'a, P: Pipeline>(&mut self, tasks: impl Iterator<Item=RenderTask<'a, P>>) {
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: None,
        });

        let frame = self.swap_chain.get_next_texture().unwrap();

        for task in tasks {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[
                    wgpu::RenderPassColorAttachmentDescriptor {
                        attachment: &frame.view,
                        resolve_target: None,
                        load_op: wgpu::LoadOp::Clear,
                        store_op: wgpu::StoreOp::Store,
                        clear_color: wgpu::Color::TRANSPARENT,
                    },
                ],
                depth_stencil_attachment: None,
            });
            pass.set_pipeline(&task.pipeline.pipeline);
            pass.set_bind_group(0, task.bindings.group(), &[]);
            pass.set_vertex_buffer(0, &task.model.buf, 0, (task.model.vert_count * std::mem::size_of::<P::Vertex>()) as wgpu::BufferAddress);
            pass.draw(0..task.model.vert_count as u32, 0..1);
        }

        self.queue.submit(&[encoder.finish()]);
    }

    pub fn render<'a, P: Pipeline>(&mut self, task: RenderTask<'a, P>) {
        self.render_batch(std::iter::once(task));
    }
}

pub trait VertexAttribute {
    fn desc(&self) -> (wgpu::VertexFormat, usize);
}

impl VertexAttribute for [f32; 3] {
    fn desc(&self) -> (wgpu::VertexFormat, usize) {
        (wgpu::VertexFormat::Float3, std::mem::size_of::<Self>())
    }
}
impl VertexAttribute for [f32; 2] {
    fn desc(&self) -> (wgpu::VertexFormat, usize) {
        (wgpu::VertexFormat::Float2, std::mem::size_of::<Self>())
    }
}

pub trait Vertex: Copy + Default + 'static {
    fn attr_desc(&self, index: usize) -> Option<(wgpu::VertexFormat, usize)>;
    fn desc<R, F: FnOnce(wgpu::VertexBufferDescriptor) -> R>(&self, f: F) -> R {
        let mut offset = 0;
        let mut attrs = Vec::new();
        for i in 0.. {
            let (format, sz) = match self.attr_desc(i) {
                Some(desc) => desc,
                None => break,
            };
            attrs.push(wgpu::VertexAttributeDescriptor {
                offset,
                shader_location: i as u32,
                format,
            });
            offset += sz as wgpu::BufferAddress;
        }
        f(wgpu::VertexBufferDescriptor {
            stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::InputStepMode::Vertex,
            attributes: &attrs,
        })
    }
}

pub trait Pipeline: 'static {
    type Vertex: Vertex;
    type BindingLayout: BindingLayout;
}

pub trait Binding {
    fn binding(&self, index: u32) -> wgpu::Binding;
    fn layout(index: u32) -> wgpu::BindGroupLayoutEntry;
}

pub struct Uniforms<U> {
    buf: wgpu::Buffer,
    _phantom: PhantomData<U>,
}

impl<U> Binding for Uniforms<U> {
    fn binding(&self, index: u32) -> wgpu::Binding {
        wgpu::Binding {
            binding: index,
            resource: wgpu::BindingResource::Buffer {
                buffer: &self.buf,
                range: 0..std::mem::size_of::<U>() as wgpu::BufferAddress,
            },
        }
    }

    fn layout(index: u32) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding: index,
            visibility: wgpu::ShaderStage::all(),
            ty: wgpu::BindingType::UniformBuffer {
                dynamic: false,
            },
        }
    }
}

pub struct Texture {
    tex: wgpu::Texture,
    view: wgpu::TextureView,
}

impl Binding for Texture {
    fn binding(&self, index: u32) -> wgpu::Binding {
        wgpu::Binding {
            binding: index,
            resource: wgpu::BindingResource::TextureView(&self.view),
        }
    }

    fn layout(index: u32) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding: index,
            visibility: wgpu::ShaderStage::all(),
            ty: wgpu::BindingType::SampledTexture {
                dimension: wgpu::TextureViewDimension::D2,
                component_type: wgpu::TextureComponentType::Float,
                multisampled: false,
            },
        }
    }
}

impl<'a, T: Binding> Binding for &'a T {
    fn binding(&self, index: u32) -> wgpu::Binding { (*self).binding(index) }
    fn layout(index: u32) -> wgpu::BindGroupLayoutEntry { T::layout(index) }
}

pub trait BindingLayout {
    fn do_for_group<R, F: FnOnce(wgpu::BindGroupDescriptor) -> R>(&self, layout: &wgpu::BindGroupLayout, f: F) -> R;
    fn do_for_layout<R, F: FnOnce(wgpu::BindGroupLayoutDescriptor) -> R>(f: F) -> R;
}

impl<A: Binding> BindingLayout for (A,) {
    fn do_for_group<R, F: FnOnce(wgpu::BindGroupDescriptor) -> R>(&self, layout: &wgpu::BindGroupLayout, f: F) -> R {
        f(wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            bindings: &[self.0.binding(0)],
        })
    }

    fn do_for_layout<R, F: FnOnce(wgpu::BindGroupLayoutDescriptor) -> R>(f: F) -> R {
        f(wgpu::BindGroupLayoutDescriptor {
            label: None,
            bindings: &[A::layout(0)],
        })
    }
}

pub struct Bindings<L: BindingLayout> {
    group: wgpu::BindGroup,
    _phantom: PhantomData<L>,
}

impl<L: BindingLayout> Bindings<L> {
    pub fn group(&self) -> &wgpu::BindGroup {
        &self.group
    }
}

pub struct Mesh<V: Vertex> {
    verts: Vec<V>,
}

impl<V: Vertex> Mesh<V> {
    pub fn push(&mut self, vert: V) {
        self.verts.push(vert);
    }
}

impl<V: Vertex> Default for Mesh<V> {
    fn default() -> Self {
        Self { verts: Vec::new() }
    }
}

impl<V: Vertex> FromIterator<V> for Mesh<V> {
    fn from_iter<I: IntoIterator<Item=V>>(iter: I) -> Self {
        let mut mesh = Self::default();
        for vert in iter {
            mesh.push(vert);
        }
        mesh
    }
}

pub struct Model<V: Vertex> {
    buf: wgpu::Buffer,
    vert_count: usize,
    _phantom: PhantomData<V>,
}

pub trait Stage {
    const STAGE: glsl_to_spirv::ShaderType;
}

pub struct Vert;
pub struct Frag;

impl Stage for Vert {
    const STAGE: glsl_to_spirv::ShaderType = glsl_to_spirv::ShaderType::Vertex;
}
impl Stage for Frag {
    const STAGE: glsl_to_spirv::ShaderType = glsl_to_spirv::ShaderType::Fragment;
}

pub struct Shader<P: Pipeline, S: Stage> {
    module: wgpu::ShaderModule,
    _phantom: PhantomData<(P, S)>,
}

pub struct PipelineState<P: Pipeline> {
    pipeline: wgpu::RenderPipeline,
    binding_layout: wgpu::BindGroupLayout,
    _phantom: PhantomData<P>,
}

pub struct RenderTask<'a, P: Pipeline> {
    pub model: &'a Model<P::Vertex>,
    pub pipeline: &'a PipelineState<P>,
    pub bindings: &'a Bindings<P::BindingLayout>,
}
