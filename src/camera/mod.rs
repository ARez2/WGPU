use instant::Duration;
use wgpu::{Device, Queue, SurfaceConfiguration};
use wgpu::util::DeviceExt;

pub mod uniform;
pub mod controller;
pub mod camera_item;


pub struct Camera {
    pub camera_item: camera_item::CameraItem,
    pub projection: camera_item::Projection,
    pub camera_controller: Option<controller::CameraController>,
    pub camera_uniform: uniform::CameraUniform,
    pub camera_buffer: wgpu::Buffer,
    pub camera_bind_group: wgpu::BindGroup,
    pub camera_bind_group_layout: wgpu::BindGroupLayout,
}

impl Camera {
    pub fn new(config: &SurfaceConfiguration, device: &Device) -> Camera {
        let camera_uniform = uniform::CameraUniform::new();
        let camera_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Camera Buffer"),
                contents: bytemuck::cast_slice(&[camera_uniform]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            }
        );
        let camera_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }
            ],
            label: Some("camera_bind_group_layout"),
        });
        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &camera_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buffer.as_entire_binding(),
                }
            ],
            label: Some("camera_bind_group"),
        });

        let temp_camera_item = camera_item::CameraItem::new((0.0, 5.0, 10.0), cgmath::Deg(-90.0), cgmath::Deg(-20.0));
        let projection = camera_item::Projection::new(config.width, config.height, cgmath::Deg(60.0), 0.1, 500.0);
        
        let mut cam = Camera {
            camera_item: temp_camera_item,
            projection,
            camera_controller: None,
            camera_uniform: uniform::CameraUniform::new(),
            camera_buffer,
            camera_bind_group,
            camera_bind_group_layout,
        };
        let camera_item = camera_item::CameraItem::new((0.0, 5.0, 10.0), cgmath::Deg(-90.0), cgmath::Deg(-20.0));
        cam.camera_uniform.update_view_proj(&camera_item, &cam.projection);
        cam.camera_item = camera_item;
        cam
    }


    pub fn update(&mut self, queue: &Queue, delta: Duration) {
        if self.camera_controller.is_some() {
            let controller = self.camera_controller.as_mut().unwrap();
            controller.update_camera(&mut self.camera_item, delta);
        };
        self.camera_uniform.update_view_proj(&self.camera_item, &self.projection);
        queue.write_buffer(&self.camera_buffer, 0, bytemuck::cast_slice(&[self.camera_uniform]));
    }
}