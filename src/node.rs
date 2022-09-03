use crate::input::Input;


pub trait Node {
    fn _process(&mut self, input: &Input, delta: instant::Duration);
}