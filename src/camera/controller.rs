use cgmath::{Vector3, Rad, InnerSpace};
use winit::{event::*, dpi::PhysicalPosition};
use instant::Duration;
use std::f32::consts::FRAC_PI_2;

use crate::{node::Node, input::Input};

use super::{camera_item};

const SAFE_FRAC_PI_2: f32 = FRAC_PI_2 - 0.0001;


#[derive(Debug)]
pub struct CameraController {
    amount_left: f32,
    amount_right: f32,
    amount_forward: f32,
    amount_backward: f32,
    amount_up: f32,
    amount_down: f32,
    rotate_horizontal: f32,
    rotate_vertical: f32,
    scroll: f32,
    speed: f32,
    sensitivity: f32,
}

impl CameraController {
    pub fn new(speed: f32, sensitivity: f32) -> Self {
        Self {
            amount_left: 0.0,
            amount_right: 0.0,
            amount_forward: 0.0,
            amount_backward: 0.0,
            amount_up: 0.0,
            amount_down: 0.0,
            rotate_horizontal: 0.0,
            rotate_vertical: 0.0,
            scroll: 0.0,
            speed,
            sensitivity,
        }
    }

    pub fn process_keyboard(&mut self, key: VirtualKeyCode, state: ElementState) -> bool{
        let amount = if state == ElementState::Pressed { 1.0 } else { 0.0 };
        match key {
            VirtualKeyCode::W | VirtualKeyCode::Up => {
                self.amount_forward = amount;
                true
            }
            VirtualKeyCode::S | VirtualKeyCode::Down => {
                self.amount_backward = amount;
                true
            }
            VirtualKeyCode::A | VirtualKeyCode::Left => {
                self.amount_left = amount;
                true
            }
            VirtualKeyCode::D | VirtualKeyCode::Right => {
                self.amount_right = amount;
                true
            }
            VirtualKeyCode::Space => {
                self.amount_up = amount;
                true
            }
            VirtualKeyCode::LShift => {
                self.amount_down = amount;
                true
            }
            _ => false,
        }
    }

    pub fn process_mouse(&mut self, mouse_dx: f64, mouse_dy: f64) {
        self.rotate_horizontal = mouse_dx as f32;
        self.rotate_vertical = mouse_dy as f32;
    }

    pub fn process_scroll(&mut self, delta: &MouseScrollDelta) {
        self.scroll = -match delta {
            // I'm assuming a line is about 100 pixels
            MouseScrollDelta::LineDelta(_, scroll) => scroll * 100.0,
            MouseScrollDelta::PixelDelta(PhysicalPosition {
                y: scroll,
                ..
            }) => *scroll as f32,
        };
    }

    pub fn update_camera(&mut self, camera_item: &mut camera_item::CameraItem, dt: Duration) {
        let dt = dt.as_secs_f32();

        // Move forward/backward and left/right
        let (yaw_sin, yaw_cos) = camera_item.yaw.0.sin_cos();
        let forward = Vector3::new(yaw_cos, 0.0, yaw_sin).normalize();
        let right = Vector3::new(-yaw_sin, 0.0, yaw_cos).normalize();
        camera_item.position += forward * (self.amount_forward - self.amount_backward) * self.speed * dt;
        camera_item.position += right * (self.amount_right - self.amount_left) * self.speed * dt;

        // Move in/out (aka. "zoom")
        // Note: this isn't an actual zoom. The camera's position
        // changes when zooming. I've added this to make it easier
        // to get closer to an object you want to focus on.
        let (pitch_sin, pitch_cos) = camera_item.pitch.0.sin_cos();
        let scrollward = Vector3::new(pitch_cos * yaw_cos, pitch_sin, pitch_cos * yaw_sin).normalize();
        camera_item.position += scrollward * self.scroll * self.speed * self.sensitivity * dt;
        self.scroll = 0.0;

        // Move up/down. Since we don't use roll, we can just
        // modify the y coordinate directly.
        camera_item.position.y += (self.amount_up - self.amount_down) * self.speed * dt;

        // Rotate
        camera_item.yaw += Rad(self.rotate_horizontal) * self.sensitivity * dt;
        camera_item.pitch += Rad(-self.rotate_vertical) * self.sensitivity * dt;

        // If process_mouse isn't called every frame, these values
        // will not get set to zero, and the camera_item will rotate
        // when moving in a non cardinal direction.
        self.rotate_horizontal = 0.0;
        self.rotate_vertical = 0.0;

        // Keep the camera's angle from going too high/low.
        if camera_item.pitch < -Rad(SAFE_FRAC_PI_2) {
            camera_item.pitch = -Rad(SAFE_FRAC_PI_2);
        } else if camera_item.pitch > Rad(SAFE_FRAC_PI_2) {
            camera_item.pitch = Rad(SAFE_FRAC_PI_2);
        }
    }
}


impl Node for CameraController {
    fn _process(&mut self, input: &Input, delta: instant::Duration) {
        self.amount_forward = input.is_action_pressed("move_forward") as i32 as f32;
        self.amount_backward = input.is_action_pressed("move_backward") as i32 as f32;
        self.amount_left = input.is_action_pressed("move_left") as i32 as f32;
        self.amount_right = input.is_action_pressed("move_right") as i32 as f32;
        self.amount_up = input.is_action_pressed("move_up") as i32 as f32;
        self.amount_down = input.is_action_pressed("move_down") as i32 as f32;
    }
}