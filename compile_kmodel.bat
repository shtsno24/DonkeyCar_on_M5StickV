python Freeze.py
python Convert_to_tflite.py
.\ncc.exe compile Model_Original.tflite Model_Original.kmodel -i tflite -o kmodel --dataset ./TestImg/ --dataset-format image --inference-type uint8 --input-mean 0 --input-std 1 --dump-ir --input-type uint8 --max-allocator-solve-secs 120 --calibrate-method l2 --dump-weights-range --weights-quantize-threshold 1024 --output-quantize-threshold 4096