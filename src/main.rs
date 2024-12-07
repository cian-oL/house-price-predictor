use anyhow::Result;
use polars::prelude::*;
use std::fs::File;
use std::io::Write;

const DATASET_URL: &str =
    "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv";
const OUTPUT_FILE: &str = "./data/boston_housing.csv";

fn download_dataset() -> Result<()> {
    println!("Downloading Boston Housing dataset...");

    let response = reqwest::blocking::get(DATASET_URL)?;
    let data = response.text()?;

    let mut file = File::create(OUTPUT_FILE)?;
    file.write_all(data.as_bytes())?;

    println!("Dataset downloaded successfully to {}", OUTPUT_FILE);
    Ok(())
}

fn load_csv_file(file_path: &str) -> Result<DataFrame> {
    let df = LazyCsvReader::new(file_path).finish()?.collect()?;

    println!("Loaded {} rows and {} columns", df.height(), df.width());
    print!("First 5 rows:\n{:#?}\n", df.head(Some(5)));

    Ok(df)
}

fn main() -> Result<()> {
    // Download the dataset to disk
    download_dataset()?;

    // Load file into memory
    let _df = load_csv_file(OUTPUT_FILE)?;
    // Prepare the data
    // Train an XGBoost model
    // Push to S3 bucket

    Ok(())
}
