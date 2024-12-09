use actix_web::{get, post, web, App, HttpResponse, HttpServer, Responder};
use std::io::Result;

#[get("/health")]
async fn health() -> impl Responder {
    HttpResponse::Ok().body("Health check OK")
}

#[actix_web::main]
async fn main() -> Result<()> {
    HttpServer::new(|| App::new().service(health))
        .bind(("127.0.0.1", 8080))?
        .run()
        .await
}
