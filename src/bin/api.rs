use actix_web::{get, post, web, App, HttpResponse, HttpServer, Responder};
use serde::Deserialize;
use std::io::Result;

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

#[get("/health")]
async fn health() -> impl Responder {
    HttpResponse::Ok().body("Health check OK")
}

#[post("/predict")]
async fn predict(payload: web::Json<PredictRequest>) -> impl Responder {
    println!("Received prediction request: {:#?}", payload);

    HttpResponse::Ok().body("Prediction OK")
}

#[actix_web::main]
async fn main() -> Result<()> {
    println!("Starting API server...");

    HttpServer::new(|| App::new().service(health).service(predict))
        .bind(("127.0.0.1", 8080))?
        .run()
        .await
}
