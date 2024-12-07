use anyhow::Result;

use house_price_predictor::{download_dataset, load_csv_file};

const DATASET_URL: &str =
    "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv";
const OUTPUT_FILE: &str = "./data/boston_housing.csv";

fn main() -> Result<()> {
    // Download the dataset to disk
    download_dataset(DATASET_URL, OUTPUT_FILE)?;

    // Load file into memory
    let _df = load_csv_file(OUTPUT_FILE)?;

    // Prepare the data by random splitting inro train and test sets

    // Train an XGBoost model
    // Push to S3 bucket

    Ok(())
}
