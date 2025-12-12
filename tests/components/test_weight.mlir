module {
  func.func @host() {
    %w1 = cherry.weight "/home/nx/ycy/pb/cherry/utils/stories110M/final_rms_norm.bin" shape [768] type f32 : !cherry.cherry_tensor<[768xf32]>

    cherry.print %w1 : !cherry.cherry_tensor<[768xf32]>
    return
  }
}
