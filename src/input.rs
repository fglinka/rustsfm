use crate::graph::{Keyframe, Keypoint, Graph};
use std::path::Path;
use quick_error::quick_error;
use cv::videoio::{VideoCapture, VideoCaptureTrait};
use cv::core::{Mat, UMat, Ptr, KeyPoint, no_array, DMatch, Scalar};
use cv::features2d::{ORB, ORB_ScoreType, FlannBasedMatcher};
use cv::prelude::*;
use cv::features2d::prelude::*;
use cv::flann::{KDTreeIndexParams, SearchParams};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::rc::Rc;
use std::cmp::Ordering;
use crate::checkpoints::{KeyframeEntry, ExtractionStep};

quick_error! {
    #[derive(Debug)]
    pub enum Error {
        Io(err: std::io::Error) {
            from()
            display("Input I/O error: {}", err)
        }
        Cv(err: cv::Error) {
            from()
            display("Input OpenCV error: {}", err)
        }
        Other(err: String) {
            display("Input error: {}", err)
        }
    }
}

const N_FEATURES: i32 = 500;
const SCALE_FACTOR: f32 = 1.2;
const N_LEVELS: i32 = 8;
const EDGE_THRESHOLD: i32 = 31;
const FIRST_LEVEL: i32 = 0;
const WTA_K: i32 = 2;
const SCORE_TYPE: ORB_ScoreType = ORB_ScoreType::HARRIS_SCORE;
const PATH_SIZE: i32 = 31;
const FAST_THRESHOLD: i32 = 20;
fn create_orb() -> Result<Ptr<dyn ORB>, Error> {
    <dyn ORB>::create(N_FEATURES,
        SCALE_FACTOR,
        N_LEVELS,
        EDGE_THRESHOLD,
        FIRST_LEVEL,
        WTA_K,
        SCORE_TYPE,
        PATH_SIZE,
        FAST_THRESHOLD).map_err(| e | Error::from(e))
}

fn process_frame(frame: &Mat, frame_index: u64, all_descriptors: &mut Mat,
    orb: &mut Ptr<dyn ORB>, debug: bool) -> Result<KeyframeEntry, Error> {
    // Set up structures to keep keypoints and descriptors
    let mut keypoints = cv::core::Vector::new();
    let mut descriptors = Mat::default();
    // Find and extract keypoints
    orb.detect_and_compute(
        frame,
        &no_array(),
        &mut keypoints,
        &mut descriptors,
        false)?;
    
    // Find start and end row indices that the descriptors submatrix will have in
    // all_descriptors
    let descriptors_start = all_descriptors.rows();
    let descriptors_end = descriptors_start + descriptors.rows();
    // Append descriptors
    all_descriptors.push_back(&descriptors)?;

    // show image if debug
    if debug {
        let mut debug_frame = Mat::default();
        cv::features2d::draw_keypoints(&frame, &keypoints, &mut debug_frame,
            Scalar::new(0.0, 1.0, 0.0, 1.0), cv::features2d::DrawMatchesFlags::DEFAULT)?;
        cv::highgui::imshow(&"debug", &debug_frame)?;
        cv::highgui::wait_key(1)?;
    }

    Ok(KeyframeEntry::new(frame_index, keypoints.to_vec(), descriptors_start, descriptors_end))
}

//fn process_descriptors(descriptors: &Mat, keypoint_entries: &Vec<KeyframeEntry>) -> Result<Graph, Error> {
    //// Build and train matcher
    //let mut matcher = FlannBasedMatcher::create()?;
    //FlannBasedMatcherTrait::add(&mut matcher, descriptors)?;
    //FlannBasedMatcherTrait::train(&mut matcher)?;

    //let max_distance = 10;

    //let bla: Vec<Result<Rc<Keyframe>, Error>> = keypoint_entries.drain(..).map(| entry | {
        //// Get submatrix for descriptors
        //let entry_descriptors =
            //descriptors.row_range(&entry.descriptor_rows())?;
        //// Build mask which excludes the keypoints in the current frame from the match
        //let num_descriptors = entry.descriptors_end() - entry.descriptors_start();
        //let mut mask: Mat = Mat::ones(num_descriptors, 1, cv::core::CV_8U)?.to_mat()?;
        //for i in entry.descriptors_start()..entry.descriptors_end() {
            //*mask.at_mut::<u8>(i)? = 0;
        //}
        //// Perform match 
        //let matches: cv::core::Vector<DMatch> = cv::core::Vector::default();
        //matcher.match_(&entry_descriptors, &mut matches, &mask)?;

        //// This is a map which contains the keyframe entry for each keypoint if it was relocated
        //// elsewhere.
        //let match_map: HashMap<i32, &KeyframeEntry> = matches.to_vec()
            //.drain(..)
            //// We remove all matches which are off
            //.filter(| m | (m.distance as i32) <= max_distance)
            //// Then we use a binary search to find the matching keyframe
            //.filter_map(| m | {
                //let kf_idx = keypoint_entries.binary_search_by(| kf | {
                    //if m.train_idx >= kf.descriptors_start() {
                        //if m.train_idx < kf.descriptors_end() { Ordering::Equal } else { Ordering::Less }
                    //} else {
                        //Ordering::Greater
                    //}
                //});
                //kf_idx.ok().map(|idx| (m.query_idx, keypoint_entries.get(idx).unwrap()))
            //})
            //.collect();


        //Ok(Keyframe::new(Vec::new(), 0))
    //}).collect();

    //panic!("Not implemented");
//}

pub fn read_from_video<P: AsRef<Path>>(video_path: P, debug: bool) -> Result<ExtractionStep, Error> {
    // Convert path to str, may fail if path node unicode
    let video_path_str = video_path.as_ref().to_str()
        .ok_or_else(| | Error::Other("Path not unicode".to_string()))?;
    let mut video_capture = VideoCapture::from_file(video_path_str,
        cv::videoio::CAP_FFMPEG)?;
    // Create ORB structure
    let mut orb = create_orb()?;
    // Create frame variable to extract frames to, may help reusing memory
    let mut frame = Mat::default();
    let mut frame_id: u64 = 0;
    // Create vectors to store recorded data in
    let mut keyframe_entries: Vec<KeyframeEntry> = Vec::new();
    let mut descriptors = Mat::default();
    // Read frames while we have some
    while video_capture.read(&mut frame)? {
        let frame_entry = process_frame(&frame, frame_id, &mut descriptors, &mut orb, debug)?;
        // Append keyframe entry
        keyframe_entries.push(frame_entry);
        // Increment frame id
        frame_id = frame_id + 1;
    }

    Ok(ExtractionStep::new(descriptors, keyframe_entries))
}
