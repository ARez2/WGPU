use node::Node;
use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop}, window::CursorGrabMode,
};
use winit::{window::Window};

mod texture;
mod camera;
mod model;
mod resources;
mod input;
mod node;
mod renderer;
use renderer::Renderer;



pub struct State {
    pub renderer: Renderer,
    pub input: input::Input,
    pub model: model::Model,
}

impl State {
    // Creating some of the wgpu types requires async code
    pub async fn new(window: &Window) -> Self {
        let renderer = Renderer::new(window).await;
        
        let mut input = input::Input::new();
        input.add_action("move_left", Some(VirtualKeyCode::A), None);
        input.add_action("move_right", Some(VirtualKeyCode::D), None);
        input.add_action("move_forward", Some(VirtualKeyCode::W), None);
        input.add_action("move_backward", Some(VirtualKeyCode::S), None);
        input.add_action("move_up", Some(VirtualKeyCode::Space), None);
        input.add_action("move_down", Some(VirtualKeyCode::LShift), None);
        input.add_action("LMB", None, Some(MouseButton::Left));
        
        let model = resources::load_model(
            "cube.obj",
            &renderer
        ).await.unwrap();

        Self {
            renderer,
            input,
            model,
        }
    }
    

    /// indicate whether an event has been fully processed.
    /// If the method returns true, the main loop won't process the event any further.
    pub fn process_input(&mut self, event: &WindowEvent) -> bool {
        let res = self.input.update(event);
        res
    }


    pub fn process_mouse_motion(&mut self, move_delta: (f64, f64)) {
        if let Some(controller) = &mut self.renderer.camera.camera_controller {
            controller.process_mouse(move_delta.0, move_delta.1);
            // if self.input.is_action_pressed("LMB") {
            // };
        };
    }


    pub fn update(&mut self, delta: instant::Duration) {
        // TODO: Replace with scene tree (process every node)
        if let Some(controller) = &mut self.renderer.camera.camera_controller {
            controller._process(&self.input, delta);
        }
        // WindowEvent::MouseWheel { delta, .. } => {
        //     self.camera_controller.process_scroll(delta);
        self.renderer.update(delta);
    }


    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        // TODO: Replace with scene tree as parameter
        self.renderer.render(&self.model)
    }
}





//#[cfg_attr(target_arch = "wasm32", wasm_bindgen(start))]
pub async fn run() {
    env_logger::init();
    let event_loop = EventLoop::new();
    let title = env!("CARGO_PKG_NAME");
    let window = winit::window::WindowBuilder::new()
        .with_title(title)
        .build(&event_loop)
        .unwrap();
    window.set_cursor_grab(CursorGrabMode::Confined)
        .or_else(|_e| window.set_cursor_grab(CursorGrabMode::Locked))
        .unwrap();
    window.set_cursor_visible(false);

    let mut state = State::new(&window).await;
    let mut last_render_time = instant::Instant::now();

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;

        match event {
            Event::MainEventsCleared => window.request_redraw(),
            Event::DeviceEvent {
                event: DeviceEvent::MouseMotion{ delta, },
                .. // We're not using device_id currently
            } => state.process_mouse_motion(delta),
            
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == window.id() => {
                let inp = state.process_input(event);
                match event {
                    #[cfg(not(target_arch="wasm32"))]
                    WindowEvent::CloseRequested
                    | WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
                                state: ElementState::Pressed,
                                virtual_keycode: Some(VirtualKeyCode::Escape),
                                ..
                            },
                        ..
                    } => *control_flow = ControlFlow::Exit,
                    WindowEvent::Resized(physical_size) => {
                        state.renderer.resize(*physical_size);
                    }
                    WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                        state.renderer.resize(**new_inner_size);
                    }
                    _ => {}
                }
            },
            Event::RedrawRequested(window_id) if window_id == window.id() => {
                let now = instant::Instant::now();
                let delta = now - last_render_time;
                last_render_time = now;
                state.update(delta);
                match state.render() {
                    Ok(_) => {}
                    // Reconfigure the surface if lost
                    Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => state.renderer.resize(state.renderer.size),
                    // The system is out of memory, we should probably quit
                    Err(wgpu::SurfaceError::OutOfMemory) => *control_flow = ControlFlow::Exit,
                    // We're ignoring timeouts
                    Err(wgpu::SurfaceError::Timeout) => log::warn!("Surface timeout"),
                }
            },
            _ => {
                state.process_mouse_motion((0.0, 0.0));
                window.request_redraw();
            }
        }
    });
}
