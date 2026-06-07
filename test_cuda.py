from numba import cuda

def check_cuda():
    print("--- CUDA Detection ---")
    
    # 1. Check if CUDA is available at all
    is_available = cuda.is_available()
    print(f"CUDA Available: {is_available}")
    
    if not is_available:
        print("\nNumba cannot find CUDA. The nvvm.dll or CUDA_HOME is still missing.")
        return

    # 2. Print out the detected hardware and drivers
    print("\n--- GPU Details ---")
    cuda.detect()

if __name__ == "__main__":
    check_cuda()
