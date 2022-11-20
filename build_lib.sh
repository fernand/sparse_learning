clear
rm libsparse.so
nvcc -Xcompiler '-fPIC' --shared lib.cu -lcusparse -lcusparseLt -o libsparse.so