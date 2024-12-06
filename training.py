input_shape = (32, 1, 32, 32)  # batch, channel, height, width
kernel_size = 5
stride = 1
padding = 0

output_height = (input_shape[2] - kernel_size + 2 * padding) // stride + 1
output_width = (input_shape[3] - kernel_size + 2 * padding) // stride + 1

print(f"Output height: {output_height}")
print(f"Output width: {output_width}")