use anyhow::Result;
use polars::prelude::*;
use rand::prelude::*;
use std::fs::File;
use std::io::Write;
use xgboost::{parameters, Booster, DMatrix};

pub fn download_dataset(dataset_url: &str, output_file: &str) -> Result<()> {
    println!("Downloading Boston Housing dataset...");

    let response = reqwest::blocking::get(dataset_url)?;
    let data = response.text()?;

    let mut file = File::create(output_file)?;
    file.write_all(data.as_bytes())?;

    println!("Dataset downloaded successfully to {}", output_file);
    Ok(())
}

pub fn load_csv_file(file_path: &str) -> Result<DataFrame> {
    let df = LazyCsvReader::new(file_path).finish()?.collect()?;

    println!("Loaded {} rows and {} columns", df.height(), df.width());
    print!("First 5 rows:\n{:#?}\n", df.head(Some(5)));

    Ok(df)
}

pub fn split_train_test(df: &DataFrame, test_size_percent: f64) -> Result<(DataFrame, DataFrame)> {
    if test_size_percent <= 0.0 || test_size_percent >= 1.0 {
        return Err(anyhow::anyhow!(PolarsError::ComputeError(
            "test_size_percent must be between 0 and 1".into(),
        )));
    }

    // Create random index range to split data
    let mut rng = rand::thread_rng();
    let mut indices: Vec<usize> = (0..df.height()).collect();
    indices.shuffle(&mut rng);

    // Split data based on desired test percentage of df
    let test_size = (test_size_percent * df.height() as f64) as usize;
    let (test_indices, train_indices) = indices.split_at(test_size);

    // Convert to ChunkedArray<Int32Type>
    let train_indices_ca = UInt32Chunked::from_vec(
        "training",
        train_indices.iter().map(|&x| x as u32).collect(),
    );
    let test_indices_ca =
        UInt32Chunked::from_vec("testing", test_indices.iter().map(|&x| x as u32).collect());

    let train_df = df.take(&train_indices_ca)?;
    let test_df = df.take(&test_indices_ca)?;

    println!("Training set size: {}", train_df.height());
    println!("Testing set size: {}", test_df.height());

    Ok((train_df, test_df))
}

// Split into features and target
pub fn split_features_target(df: &DataFrame) -> Result<(DataFrame, DataFrame)> {
    let features = df.select([
        "crim", "zn", "indus", "chas", "nox", "rm", "age", "dis", "rad", "tax", "ptratio", "black",
        "lstat",
    ])?;

    let target = df.select(["medv"])?;

    Ok((features, target))
}

// trains the model with xgboost, evaluates on test set, saves model locally in a models directory and returns the path to the model
pub fn train_model(
    x_train_df: &DataFrame,
    y_train_df: &DataFrame,
    x_test_df: &DataFrame,
    y_test_df: &DataFrame,
) -> Result<String> {
    // Transform Polars DataFrames into 2D arrays in row-major order
    let x_train = x_train_df.to_ndarray::<Float32Type>(IndexOrder::C)?;
    let y_train = y_train_df.to_ndarray::<Float32Type>(IndexOrder::C)?;
    let x_test = x_test_df.to_ndarray::<Float32Type>(IndexOrder::C)?;
    let y_test = y_test_df.to_ndarray::<Float32Type>(IndexOrder::C)?;

    // Convert the 2D arrays into slices &[f32]
    let x_train = x_train
        .as_slice()
        .expect("Failed to convert x_train to slice - array may not be contiguous");
    let y_train = y_train
        .as_slice()
        .expect("Failed to convert y_train to slice - array may not be contiguous");
    let x_test = x_test
        .as_slice()
        .expect("Failed to convert x_test to slice - array may not be contiguous");
    let y_test = y_test
        .as_slice()
        .expect("Failed to convert y_test to slice - array may not be contiguous");

    // START HERE!
    Ok("".to_string())
}
