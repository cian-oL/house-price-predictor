use anyhow::Result;
use aws_config::{defaults, meta::region::RegionProviderChain, BehaviorVersion};
use aws_sdk_s3::Client;
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
        "crim", "zn", "indus", "chas", "nox", "rm", "age", "dis", "rad", "tax", "ptratio", "b",
        "lstat",
    ])?;

    let target = df.select(["medv"])?;

    Ok((features, target))
}

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

    // Convert training and testing sets to XGBoost DMatrix objects for evaluation
    let mut dtrain = DMatrix::from_dense(&x_train.clone().into_raw_vec(), x_train.nrows())?;
    let mut dtest = DMatrix::from_dense(&x_test.clone().into_raw_vec(), x_test.nrows())?;

    dtrain.set_labels(
        y_train
            .as_slice()
            .ok_or_else(|| anyhow::anyhow!("Training array not contiguous"))?,
    )?;

    dtest.set_labels(
        y_test
            .as_slice()
            .ok_or_else(|| anyhow::anyhow!("Testing array not contiguous"))?,
    )?;

    let evaluation_sets = &[(&dtrain, "train"), (&dtest, "test")];

    // Specify overall training setup
    let training_params = parameters::TrainingParametersBuilder::default()
        .dtrain(&dtrain)
        .evaluation_sets(Some(evaluation_sets))
        .build()
        .unwrap();

    // Train model, and print evaluation data
    let bst = Booster::train(&training_params).unwrap();
    println!("Test {:?}", bst.predict(&dtest).unwrap());

    // Save model to disk
    let model_path = "./output/models/model.bin";
    bst.save(model_path)?;
    println!("Model saved to {}", model_path);

    Ok(model_path.to_string())
}

// pushes the given file to an S3 bucket
pub async fn push_to_s3_bucket(path_to_model: &str) -> Result<()> {
    // Create an AWS S3 client so I can talk to the S3 service
    let region_provider = RegionProviderChain::default_provider().or_else("us-east-1");

    let config = defaults(BehaviorVersion::latest())
        .region(region_provider)
        .load()
        .await;

    let client = Client::new(&config);

    // Load the model file into memory
    let model_file_bytes = std::fs::read(path_to_model)?;
    // Upload the model file to the S3 bucket
    // TODO: make this value a parameter to this function
    let bucket_name = "house-price-predictor";
    let key = "boston-housing-model.bin";
    let _result = client
        .put_object()
        .bucket(bucket_name)
        .key(key)
        .body(model_file_bytes.into())
        .send()
        .await?;
    Ok(())
}
