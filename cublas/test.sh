nm=(128 256 512 1024 2048 4096 8192 16384 32768)
nn=(128 256 512 1024 2048 4096 8192 16384 32768)

m=32768
n=32768
k=32768

for indexm in "${!nm[@]}"; do
    for indexn in "${!nn[@]}"; do
        ./cublas_gemm_batched_float $m $n $k ${nm[$indexm]} ${nn[$indexn]} | tee -a test_float.log
    done
done
