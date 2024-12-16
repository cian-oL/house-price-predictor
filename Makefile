# command with sample for boston housing dataset
run-train-dev:
	cargo run --bin train -- \
		--dataset-url https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv \
		--dataset-file-name boston_housing.csv \
		--bucket-name house-price-predictor-rust \
		--key boston-housing-model.bin

run-api-dev:
	cargo run --bin api

# Test the API to return a successful post request with one line of data
request-predict-dev:
	curl -X POST http://localhost:8080/predict \
		-H "Content-Type: application/json" \
		-d '{ \
			"crim": 0.00632, \
			"zn": 18.0, \
			"indus": 2.31, \
			"chas": 0, \
			"nox": 0.538, \
			"rm": 6.575, \
			"age": 65.2, \
			"dis": 4.0900, \
			"rad": 1, \
			"tax": 296, \
			"ptratio": 15.3, \
			"b": 396.90, \
			"lstat": 4.98 \
		}'
