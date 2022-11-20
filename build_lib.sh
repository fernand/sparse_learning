clear
rm libsparse.so
# TODO do sm_86 gencode in 3 steps per https://malithjayaweera.com/2019/03/compilation-linking-cuda-c/ and bitsandbytes
nvcc -Xcompiler '-fPIC' --shared lib.cu -lcudart -lcusparse -lcusparseLt -o libsparse.so