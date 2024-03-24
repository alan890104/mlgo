export PROGRAM_PATH="./mlgo/examples/mnist_mips/mlgo.bin"
export MODEL_PATH="./mlgo/examples/mnist/models/mnist/ggml-model-small-f32-big-endian.bin"
export DATA_PATH="./mlgo/examples/mnist/models/mnist/input_7"
mlvm/mlvm --basedir=/tmp/cannon_fault --program="$PROGRAM_PATH" --model="$MODEL_PATH" --data="$DATA_PATH" --mipsVMCompatible
