use actix_web::{get, post, App, HttpResponse, HttpServer, Responder};
use serde::Deserialize;
use std::io::Result;

#[derive(Deserialize)]
struct PredictRequest {
    #[serde(rename = "CRIM")]
    crime_rate: f64,
    #[serde(rename = "ZN")]
    prop_residential_land: f64,
    #[serde(rename = "INDUS")]
    prop_non_retail_business_acres: f64,
    #[serde(rename = "CHAS")]
    charles_river_dummy: i8,
    #[serde(rename = "NOX")]
    nitric_oxide_concentration: f64,
    #[serde(rename = "RM")]
    avg_rooms_per_dwelling: f64,
    #[serde(rename = "AGE")]
    prop_owner_occupied_homes_built_before_1940: f64,
    #[serde(rename = "DIS")]
    weighted_distances_to_five_boston_employment_centers: f64,
    #[serde(rename = "RAD")]
    index_of_accessibility_to_radial_highways: f64,
    #[serde(rename = "TAX")]
    full_value_property_tax_rate_per_10k_dollars: f64,
    #[serde(rename = "PTRATIO")]
    pupil_teacher_ratio: f64,
    #[serde(rename = "B")]
    prop_black_population: f64,
    #[serde(rename = "LSTAT")]
    lower_status_of_population: f64,
}

#[get("/health")]
async fn health() -> impl Responder {
    HttpResponse::Ok().body("Health check OK")
}

#[post("/predict")]
async fn predict() -> impl Responder {
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
