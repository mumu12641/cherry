module {
  func.func @host() {
    %wo = cherry.weight "/home/nx/ycy/pb/cherry/utils/stories110M/layers_wo.bin" shape[12, 768, 768] type f32 -> !cherry.cherry_tensor<[12x768x768xf32]>
    %wo_transpose = cherry.transpose %wo perm [1, 2, 0] : (!cherry.cherry_tensor<[12x768x768xf32]>) -> !cherry.cherry_tensor<[768x768x12xf32]>
    cherry.print %wo_transpose : !cherry.cherry_tensor<[768x768x12xf32]>
    return
  }
}
