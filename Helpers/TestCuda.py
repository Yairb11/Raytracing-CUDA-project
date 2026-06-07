from numba import cuda

def check_cuda():
    print("--- CUDA Detection ---")
    is_available = cuda.is_available()
    print(f"CUDA Available: {is_available}")
    if not is_available:
        print("\nNumba cannot find CUDA. The nvvm.dll or CUDA_HOME is still missing.")
        return
    print("\n--- GPU Details ---")
    cuda.detect()

if __name__ == "__main__":
    check_cuda()
'''
for i1 in range(3):
    color[i1] = triangles_color[hit_index, i1] * ambient_intensity
'''