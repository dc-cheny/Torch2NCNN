export ONNX_NAME="mobilenet_arcface_epoch20_230807"
export GENERATE_ROOT_PATH="C:\\worksp\\xxcy\\Torch2NCNN\\data\\cls_models\\${ONNX_NAME}\\"
mkdir ${GENERATE_ROOT_PATH}

export ONNX_PATH="${GENERATE_ROOT_PATH}${ONNX_NAME}.onnx"
export SIM_FILE="${GENERATE_ROOT_PATH}${ONNX_NAME}-sim.onnx"
/c/Users/Crty/anaconda3/envs/torch1.9/python -m onnxsim $ONNX_PATH $SIM_FILE

export SIM_PARAM="${GENERATE_ROOT_PATH}${ONNX_NAME}-sim.param"
export SIM_BIN="${GENERATE_ROOT_PATH}${ONNX_NAME}-sim.bin"
C:\\worksp\\xxcy\\Torch2NCNN\\ncnn-20230517-windows-vs2022\\x64\\bin\\onnx2ncnn.exe $SIM_FILE $SIM_PARAM $SIM_BIN

export IDH="${GENERATE_ROOT_PATH}${ONNX_NAME}-sim.id.h"
export MEMH="${GENERATE_ROOT_PATH}${ONNX_NAME}-sim.mem.h"
C:\\worksp\\xxcy\\Torch2NCNN\\ncnn-20230517-windows-vs2022\\x64\\bin\\ncnn2mem.exe $SIM_PARAM $SIM_BIN $IDH $MEMH
