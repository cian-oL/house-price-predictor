use anyhow::Result;
use tokio::runtime::Runtime;

use house_price_predictor::*;

const DATASET_URL: &str =
    "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv";
const OUTPUT_FILE: &str = "./data/boston_housing.csv";

fn main() -> Result<()> {
    // Download the dataset to disk
    download_dataset(DATASET_URL, OUTPUT_FILE)?;

    // Load file into memory
    let df = load_csv_file(OUTPUT_FILE)?;

    // Prepare the data by random splitting inro train and test sets
    let (train_df, test_df) = split_train_test(&df, 0.2)?;

    // Split into features and target
    let (x_train_df, y_train_df) = split_features_target(&train_df)?;
    let (x_test_df, y_test_df) = split_features_target(&test_df)?;

    // Train the model
    let path_to_model = train_model(&x_train_df, &y_train_df, &x_test_df, &y_test_df)?;

    // Push to S3 bucket
    let runtime = Runtime::new()?;
    runtime.block_on(push_to_s3_bucket(&path_to_model))?;
    println!("Model pushed to S3 bucket");

    Ok(())
}
