use actix_web::{get, post, web, App, HttpResponse, HttpServer, Responder};
use clap::Parser;
use serde::Deserialize;
use std::{io::Result, sync::Arc};
use xgboost::{Booster, DMatrix};

use house_price_predictor::modules::{
    aws::download_model_from_s3_bucket, model::load_model_to_memory,
};

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Bucket name
    #[arg(short, long)]
    bucket_name: String,

    /// Key (for S3 file)
    #[arg(short = 'k', long = "key")]
    s3_file_key: String,
}

struct AppState {
    model: Arc<Booster>,
}

#[derive(Debug, Deserialize)]
struct PredictRequest {
    #[serde(rename = "crim")]
    crime_rate: f64,

    #[serde(rename = "zn")]
    large_zones_percent: f64,

    #[serde(rename = "indus")]
    non_retail_business_acres: f64,

    #[serde(rename = "chas")]
    charles_river_dummy: i8,

    #[serde(rename = "nox")]
    nitric_oxide_concentration: f64,

    #[serde(rename = "rm")]
    avg_rooms_per_dwelling: f64,

    #[serde(rename = "age")]
    homes_pre_1940_percent: f64,

    #[serde(rename = "dis")]
    employment_centers_weighted_distance: f64,

    #[serde(rename = "rad")]
    highway_accessibility_index: f64,

    #[serde(rename = "tax")]
    property_tax_rate: f64,

    #[serde(rename = "ptratio")]
    pupil_teacher_ratio: f64,

    #[serde(rename = "b")]
    black_population: f64,

    #[serde(rename = "lstat")]
    lower_status_percent: f64,
}

const MODEL_PATH: &str = "./input/models/downloaded-model.bin";

#[get("/health")]
async fn health() -> impl Responder {
    HttpResponse::Ok().body("Health check OK")
}

#[post("/predict")]
async fn predict(payload: web::Json<PredictRequest>, data: web::Data<AppState>) -> impl Responder {
    println!("Received prediction request: {:#?}", payload);

    // // confirm model is available as part of the app state
    // let model_metadata = data.model.get_attribute_names().unwrap();
    // println!("Model metadata: {:?}", model_metadata);

    let dmatrix_features = transform_payload_to_dmatrix(&payload).unwrap();

    println!("Features sent by the client: {:?}", payload);
    HttpResponse::Ok().body("Prediction OK")
}

/// Transform a JSON payload into a DMatrix
fn transform_payload_to_dmatrix(payload: &PredictRequest) -> anyhow::Result<DMatrix> {
    // transform payload into f64 slice
    let features: Vec<f32> = [
        payload.crime_rate,
        payload.large_zones_percent,
        payload.non_retail_business_acres,
        payload.charles_river_dummy as f64,
        payload.nitric_oxide_concentration,
        payload.avg_rooms_per_dwelling,
        payload.homes_pre_1940_percent,
        payload.employment_centers_weighted_distance,
        payload.highway_accessibility_index,
        payload.property_tax_rate,
        payload.pupil_teacher_ratio,
        payload.black_population,
        payload.lower_status_percent,
    ]
    .iter()
    .map(|f| *f as f32)
    .collect();

    Ok(DMatrix::from_dense(&features, 1)?)
}

#[actix_web::main]
async fn main() -> Result<()> {
    let Args {
        bucket_name,
        s3_file_key,
    } = Args::parse();

    download_model_from_s3_bucket(&bucket_name, &s3_file_key, MODEL_PATH)
        .await
        .unwrap();

    println!("Starting API server...");

    HttpServer::new(move || {
        let bst = load_model_to_memory(MODEL_PATH).unwrap();

        // Wrap model in Arc for thread safety
        let app_state = web::Data::new(AppState {
            model: Arc::new(bst),
        });

        App::new()
            .service(health)
            .service(predict)
            .app_data(app_state)
    })
    .bind(("127.0.0.1", 8080))?
    .run()
    .await
}
