use winit::event::{VirtualKeyCode, ElementState, WindowEvent, MouseButton, KeyboardInput};


pub struct InputAction {
    pub name: String,
    pub key: Option<VirtualKeyCode>,
    pub mouse_button: Option<MouseButton>,
    pub state: ElementState,
}


pub struct Input {
    pub actions: Vec<InputAction>,
}

impl Input {
    pub fn new() -> Input {
        Input {
            actions: vec![],
        }
    }

    pub fn add_action(&mut self, name: &str, key: Option<VirtualKeyCode>, mouse_button: Option<MouseButton>) {
        let new_action = InputAction {
            name: name.to_string(),
            key,
            mouse_button,
            state: ElementState::Released,
        };
        self.actions.push(new_action);
    }

    pub fn update(&mut self, event: &WindowEvent) -> bool {
        let mut changed_something = true;

        for action in self.actions.iter_mut() {
            match event {
                WindowEvent::KeyboardInput {
                    input:
                        KeyboardInput {
                            virtual_keycode: key,
                            state,
                            ..
                        },
                    ..
                } => {
                    if action.key == *key {
                        action.state = *state;
                    };
                },
                WindowEvent::MouseInput {
                    button,
                    state,
                    ..
                } => {
                    if action.mouse_button == Some(*button) {
                        action.state = *state;
                    };
                }
                _ => changed_something = false,
            };
        };
        changed_something
    }

    pub fn is_action_pressed(&self, action_name: &str) -> bool {
        for action in self.actions.iter() {
            if action.name == action_name {
                return action.state == ElementState::Pressed;
            };
        };
        false
    }
}