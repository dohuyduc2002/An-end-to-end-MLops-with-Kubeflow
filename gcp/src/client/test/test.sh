
pytest test_dataloader.py || { echo "Dataloader test failed. stop"; exit 1; }
echo "Dataloader test passed"

pytest test_preprocess.py || { echo "Preprocess test failed. stop"; exit 1; }
echo "Preprocess test passed"

pytest test_model_inference.py || { echo "Model inference test failed. stop"; exit 1; }
echo "Model inference test passed"
