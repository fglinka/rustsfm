use na::{Vector3, DVector};
use serde::{Serialize, Deserialize};
use std::rc::{Rc, Weak};

/// A matched point with an associated feature descriptor
#[derive(Debug, Serialize, Deserialize)]
pub struct Keypoint {
    position: Vector3<f64>,
    occurs_in: Vec<Weak<Keyframe>>,
    descriptor: DVector<f64>
}

pub type KeypointPtr = Rc<Keypoint>;

#[derive(Debug, Serialize, Deserialize)]
pub struct Keyframe {
    keypoints: Vec<KeypointPtr>,
    index: u64
}

pub type KeyframePtr = Rc<Keyframe>;

#[derive(Debug, Serialize, Deserialize)]
pub struct Graph {
    keyframes: Vec<Rc<Keyframe>>
}

impl Keypoint {
    pub fn new(position: Vector3<f64>, descriptor: DVector<f64>) -> KeypointPtr {
        Rc::new(Keypoint {
            position,
            occurs_in: Vec::new(),
            descriptor
        })
    }

    pub fn add_correspondence(&mut self, keyframe: &KeyframePtr) {
        self.occurs_in.push(Rc::<Keyframe>::downgrade(&keyframe));
    }
}

impl Keyframe {
    pub fn new(keypoints: Vec<KeypointPtr>, index: u64) -> KeyframePtr {
        Rc::new(Keyframe {
            keypoints,
            index
        })
    }
}
