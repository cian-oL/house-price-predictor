use anyhow::Result;
use clap::Parser;
use tokio::runtime::Runtime;

use house_price_predictor::modules::aws::push_model_to_s3_bucket;
use house_price_predictor::modules::data::{
    download_dataset, load_csv_file, split_features_target, split_train_test,
};
use house_price_predictor::modules::model::train_model;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Dataset URL
    #[arg(short = 'd', long)]
    dataset_url: String,

    /// Dataset file name
    #[arg(short = 'f', long)]
    dataset_file_name: String,

    /// Bucket name
    #[arg(short = 'b', long)]
    bucket_name: String,

    /// Key (for S3 file)
    #[arg(short = 'k', long = "key")]
    s3_file_key: String,
}

fn main() -> Result<()> {
    let Args {
        dataset_url,
        dataset_file_name,
        bucket_name,
        s3_file_key,
    } = Args::parse();

    // Download the dataset to disk
    download_dataset(&dataset_url, &dataset_file_name)?;

    // Load file into memory
    let df = load_csv_file(&dataset_file_name)?;

    // Prepare the data by random splitting inro train and test sets
    let (train_df, test_df) = split_train_test(&df, 0.2)?;

    // Split into features and target
    let (x_train_df, y_train_df) = split_features_target(&train_df)?;
    let (x_test_df, y_test_df) = split_features_target(&test_df)?;

    // Train the model
    let path_to_model = train_model(&x_train_df, &y_train_df, &x_test_df, &y_test_df)?;

    // Push to S3 bucket
    let runtime = Runtime::new()?;
    runtime.block_on(push_model_to_s3_bucket(
        &path_to_model,
        &bucket_name,
        &s3_file_key,
    ))?;
    println!("Model pushed to S3 bucket");

    Ok(())
}
