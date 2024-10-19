use actix_web::{web, App, HttpResponse, HttpServer, Result};
use image::ImageFormat;
use ocrs::{DecodeMethod, DimOrder, ImageSource, OcrEngine, OcrEngineParams, TextLine};
use rten_tensor::prelude::*;
use rten_tensor::NdTensor;
use std::error::Error;
use lazy_static::initialize;

#[macro_use]
extern crate lazy_static;

mod models;
use models::{load_model, ModelSource};

pub fn format_text_output(text_lines: &[Option<TextLine>]) -> String {
    let lines: Vec<String> = text_lines
        .iter()
        .flatten()
        .map(|line| line.to_string())
        .collect();
    lines.join("\n")
}

lazy_static! {
    static ref OCR_ENGINE: OcrEngine = {
        println!("Loading model...");
        // Fetch and load ML models.
        let detection_model_src = ModelSource::Url(DETECTION_MODEL);
        let detection_model = load_model(detection_model_src)
            .expect("Failed to load text detection model from");

        let recognition_model_src = ModelSource::Url(RECOGNITION_MODEL);
        let recognition_model = load_model(recognition_model_src)
        .expect("Failed to load text recognition model from");

        OcrEngine::new(OcrEngineParams {
            detection_model: Some(detection_model),
            recognition_model: Some(recognition_model),
            debug: true,
            decode_method: DecodeMethod::Greedy,
            ..Default::default()
        })
        .expect("Failed to initialize engine")
    };
}

async fn process_image(payload: web::Payload) -> Result<HttpResponse> {
    let stream = payload.to_bytes().await?;
    let img = image::load_from_memory_with_format(&stream, ImageFormat::Png);

    let color_img: NdTensor<u8, 3> = match img.map(|image| {
        let image = image.into_rgb8();
        let (width, height) = image.dimensions();
        let in_chans = 3;
        NdTensor::from_data(
            [height as usize, width as usize, in_chans],
            image.into_vec(),
        )
    }) {
        Ok(tensor) => tensor,
        Err(err) => {
            eprintln!("Failed to load image: {:?}", err);
            return Ok(HttpResponse::BadRequest().body("Failed to load image"));
        }
    };

    let engine = &OCR_ENGINE;

    // Preprocess image for use with OCR engine.
    let color_img_source = ImageSource::from_tensor(color_img.view(), DimOrder::Hwc)
        .expect("Failed to create image source");
    let ocr_input = engine
        .prepare_input(color_img_source)
        .expect("Failed to prepare input");
    let word_rects = match engine.detect_words(&ocr_input) {
        Ok(rects) => rects,
        Err(err) => {
            eprintln!("Failed to detect words: {:?}", err);
            return Ok(HttpResponse::BadRequest().body("Failed to detect words"));
        }
    };
    let line_rects = engine.find_text_lines(&ocr_input, &word_rects);
    let line_texts = engine
        .recognize_text(&ocr_input, &line_rects)
        .expect("Failed to recognize text");

    let output_text = format_text_output(&line_texts);
    Ok(HttpResponse::Ok().body(output_text))
}

const DETECTION_MODEL: &str = "https://ocrs-models.s3-accelerate.amazonaws.com/text-detection.rten";
const RECOGNITION_MODEL: &str =
    "https://ocrs-models.s3-accelerate.amazonaws.com/text-recognition.rten";

#[actix_web::main]
async fn main() -> std::result::Result<(), Box<dyn Error>> {
    initialize(&OCR_ENGINE);

    println!("Starting server at http://localhost:8080");
    HttpServer::new(|| App::new().route("/process", web::post().to(process_image)))
        .bind("0.0.0.0:8080")?
        .run()
        .await
        .map_err(|e| e.into())
}
