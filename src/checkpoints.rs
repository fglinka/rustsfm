use cv::core::{Mat, KeyPoint, Range};
use cv::core::prelude::*;
use serde::ser::{Serializer, SerializeStruct};
use serde::de::{self, Deserializer, SeqAccess, MapAccess, Visitor};
use serde::{Serialize, Deserialize};
use na::{DMatrix};
use cv_convert::TryFromCv;

struct KeyPointWrapper {
    keypoint: KeyPoint
}

#[derive(Serialize, Deserialize)]
pub struct KeyframeEntry {
    frame_id: u64,
    keypoints: Vec<KeyPointWrapper>,
    descriptors_start: i32,
    descriptors_end: i32
}

#[derive(Serialize, Deserialize)]
pub struct ExtractionStep {
    descriptors: DMatrix<u8>,
    keypoint_entries: Vec<KeyframeEntry>
}

impl KeyPointWrapper {
    fn new(keypoint: KeyPoint) -> Self {
        KeyPointWrapper { keypoint }
    }
}
impl KeyframeEntry {
    pub fn new(frame_id: u64, mut keypoints: Vec<KeyPoint>, descriptors_start: i32,
        descriptors_end: i32) -> Self {
        KeyframeEntry {
            frame_id,
            //descriptors: na::DMatrix::try_from_cv(&descriptors).expect("Mat -> DMatrix conversion failed"),
            keypoints: keypoints.drain(..).map(KeyPointWrapper::new).collect(),
            descriptors_start,
            descriptors_end
        }
    }

    pub fn descriptor_rows(&self) -> Range {
        Range::new(self.descriptors_start, self.descriptors_end).
            expect("Range construction failed")
    }

    pub fn descriptors_start(&self) -> i32 {
        self.descriptors_start
    }

    pub fn descriptors_end(&self) -> i32 {
        self.descriptors_end
    }
}

impl ExtractionStep {
    pub fn new(descriptors: Mat, keyframe_entries: Vec<KeyframeEntry>) -> Self {
        ExtractionStep {
            descriptors: na::DMatrix::try_from_cv(&descriptors)
                .expect("cv::Mat -> DMatrix failed"),
            keypoint_entries: keyframe_entries
        }
    }

    pub fn keyframe_entries(&self) -> &[KeyframeEntry] {
        &self.keypoint_entries
    }
}

impl Serialize for KeyPointWrapper {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
where S: Serializer {
        let mut state = serializer.serialize_struct("KeyPoint", 7)?;
        state.serialize_field("x", &self.keypoint.pt().x)?;
        state.serialize_field("y", &self.keypoint.pt().y)?;
        state.serialize_field("size", &self.keypoint.size())?;
        state.serialize_field("angle", &self.keypoint.angle())?;
        state.serialize_field("response", &self.keypoint.response())?;
        state.serialize_field("octave", &self.keypoint.octave())?;
        state.serialize_field("class_id", &self.keypoint.class_id())?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for KeyPointWrapper {
    fn deserialize<D>(deserializer: D) -> Result<KeyPointWrapper, D::Error>
where D: Deserializer<'de>
    {
        #[derive(Deserialize)]
        #[serde(field_identifier, rename_all = "lowercase")]
        enum Field { X, Y, Size, Angle, Response, Octave, Class_Id };

        struct KeyPointWrapperVisitor;
        impl<'de> Visitor<'de> for KeyPointWrapperVisitor {
            type Value = KeyPointWrapper;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("struct KeyPointWrapper")
            }

            fn visit_seq<V>(self, mut seq: V) -> Result<KeyPointWrapper, V::Error>
            where
                V: SeqAccess<'de>
            {
                let x: f32 = seq.next_element()?
                    .ok_or_else(|| de::Error::invalid_length(0, &self))?;
                let y: f32 = seq.next_element()?
                    .ok_or_else(|| de::Error::invalid_length(1, &self))?;
                let size: f32 = seq.next_element()?
                    .ok_or_else(|| de::Error::invalid_length(2, &self))?;
                let angle: f32 = seq.next_element()?
                    .ok_or_else(|| de::Error::invalid_length(3, &self))?;
                let response: f32 = seq.next_element()?
                    .ok_or_else(|| de::Error::invalid_length(4, &self))?;
                let octave: i32 = seq.next_element()?
                    .ok_or_else(|| de::Error::invalid_length(5, &self))?;
                let class_id: i32 = seq.next_element()?
                    .ok_or_else(|| de::Error::invalid_length(0, &self))?;

                Ok(KeyPointWrapper::new(KeyPoint::new_coords(x, y, size, angle, response,
                    octave, class_id).expect("Failed to create KeyPoint")))
            }

            fn visit_map<V>(self, mut map: V) -> Result<KeyPointWrapper, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut x: Option<f32> = None;
                let mut y: Option<f32> = None;
                let mut size: Option<f32> = None;
                let mut angle: Option<f32> = None;
                let mut response: Option<f32> = None;
                let mut octave: Option<i32> = None;
                let mut class_id: Option<i32> = None;
                while let Some(key) = map.next_key()? {
                    match key {
                        Field::X => {
                            if x.is_some() {
                                return Err(de::Error::duplicate_field("x"))
                            }
                            x = Some(map.next_value()?);
                        },
                        Field::Y => {
                            if y.is_some() {
                                return Err(de::Error::duplicate_field("y"))
                            }
                            y = Some(map.next_value()?);
                        },
                        Field::Size => {
                            if size.is_some() {
                                return Err(de::Error::duplicate_field("size"))
                            }
                            size = Some(map.next_value()?);
                        },
                        Field::Angle => {
                            if angle.is_some() {
                                return Err(de::Error::duplicate_field("angle"))
                            }
                            angle = Some(map.next_value()?);
                        },
                        Field::Response => {
                            if response.is_some() {
                                return Err(de::Error::duplicate_field("response"))
                            }
                            response = Some(map.next_value()?);
                        },
                        Field::Octave => {
                            if octave.is_some() {
                                return Err(de::Error::duplicate_field("octave"))
                            }
                            octave = Some(map.next_value()?);
                        },
                        Field::Class_Id => {
                            if class_id.is_some() {
                                return Err(de::Error::duplicate_field("class_id"))
                            }
                            class_id = Some(map.next_value()?);
                        }
                    }
                }
                let x = x.ok_or_else(|| de::Error::missing_field("x"))?;
                let y = y.ok_or_else(|| de::Error::missing_field("y"))?;
                let size = size.ok_or_else(|| de::Error::missing_field("size"))?;
                let angle = angle.ok_or_else(|| de::Error::missing_field("angle"))?;
                let response = response.ok_or_else(|| de::Error::missing_field("response"))?;
                let octave = octave.ok_or_else(|| de::Error::missing_field("octave"))?;
                let class_id = class_id.ok_or_else(|| de::Error::missing_field("class_id"))?;
                Ok(KeyPointWrapper::new(KeyPoint::new_coords(x, y, size, angle, response,
                    octave, class_id).expect("Failed to create KeyPoint")))
            }
        }

        const FIELDS: &'static [&'static str] = &["x", "y", "size", "angle", "response",
            "octave", "class_id"];
        deserializer.deserialize_struct("KeyPoint", FIELDS, KeyPointWrapperVisitor)
    }
}
