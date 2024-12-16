use anyhow::Result;
use aws_config::{
    defaults, meta::region::RegionProviderChain, profile::ProfileFileCredentialsProvider,
    BehaviorVersion,
};
use aws_sdk_s3::Client;
use std::fs::File;
use std::io::Write;

// pushes the given file to an S3 bucket
pub async fn push_model_to_s3_bucket(
    path_to_model: &str,
    bucket_name: &str,
    key: &str,
) -> Result<()> {
    let client = create_s3_client().await;

    // Load the model file into memory and upload to S3
    let model_file_bytes = std::fs::read(path_to_model)?;

    let _result = client
        .put_object()
        .bucket(bucket_name)
        .key(key)
        .body(model_file_bytes.into())
        .send()
        .await?;

    Ok(())
}

pub async fn download_model_from_s3_bucket(
    bucket_name: &str,
    key: &str,
    downloaded_file_path: &str,
) -> Result<()> {
    let client = create_s3_client().await;

    // Download the model file and convert to bytes for save to disk
    let s3_obj = client
        .get_object()
        .bucket(bucket_name)
        .key(key)
        .send()
        .await?;

    let data = s3_obj.body.collect().await?;

    let mut file = File::create(downloaded_file_path)?;
    file.write_all(&data.into_bytes())?;
    println!("Model downloaded to {}", downloaded_file_path);

    Ok(())
}

async fn create_s3_client() -> Client {
    let region_provider = RegionProviderChain::default_provider().or_else("eu-west-1");

    let config = defaults(BehaviorVersion::latest())
        .credentials_provider(
            ProfileFileCredentialsProvider::builder()
                .profile_name("default")
                .build(),
        )
        .region(region_provider)
        .load()
        .await;

    Client::new(&config)
}
