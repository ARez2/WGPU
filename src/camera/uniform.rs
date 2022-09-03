use cgmath::SquareMatrix;
use super::camera_item;


#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    pub view_position: [f32; 4],
    pub view_proj: [[f32; 4]; 4],
}

impl CameraUniform {
    pub fn new() -> Self {
        Self {
            view_position: [0.0; 4],
            view_proj: cgmath::Matrix4::identity().into(),
        }
    }

    pub fn update_view_proj(&mut self, camera_item: &camera_item::CameraItem, projection: &camera_item::Projection) {
        self.view_position = camera_item.position.to_homogeneous().into();
        self.view_proj = (projection.calc_matrix() * camera_item.calc_matrix()).into();
    }
}