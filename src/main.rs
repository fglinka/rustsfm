mod graph;
mod input;
mod checkpoints;

use crate::input::read_from_video;
use std::fs::File;
use checkpoints::ExtractionStep;
use bincode;

fn main() {
    let mut extraction_step_in = File::open("extraction.bin").unwrap();
    let extraction_step: ExtractionStep = bincode::deserialize_from(&mut extraction_step_in).unwrap();
    println!("Num keyframes: {}", extraction_step.keyframe_entries().len());
    //println!("Extracting");
    //let extraction_step = read_from_video("input.mp4", true).unwrap();
    //println!("Saving");
    //let mut extraction_out = File::create("extraction.bin").unwrap();
    //bincode::serialize_into(&mut extraction_out, &extraction_step).unwrap();
}
