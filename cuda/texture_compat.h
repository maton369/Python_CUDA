// CUDA 12.0以降でのレガシーテクスチャ参照の互換性定義
// PyCUDAがextern "C"でラップする場合でも動作するように、C++リンケージを明示

#if CUDA_VERSION >= 12000
#ifdef __cplusplus
extern "C++" {
template<typename T, int Dim, enum cudaTextureReadMode ReadMode>
struct texture {
    cudaTextureObject_t texObj;
};

// tex1Dfetch の互換性ラッパー
template<typename T>
__device__ __forceinline__ T tex1Dfetch(const texture<T, 1, cudaReadModeElementType>& tex, int x) {
    return tex1Dfetch<T>(tex.texObj, x);
}
} // extern "C++"
#endif // __cplusplus
#endif // CUDA_VERSION >= 12000

