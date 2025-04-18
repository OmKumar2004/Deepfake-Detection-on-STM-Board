from PIL import Image
import numpy as np
import os

def convert_image_to_c(image_path, array_name):
    img = Image.open(image_path).resize((64, 64)).convert("RGB")
    arr = np.asarray(img, dtype=np.uint8).flatten()

    c_array = f"const uint8_t {array_name}[{len(arr)}] = {{\n"
    for i in range(0, len(arr), 12):
        line = ", ".join(map(str, arr[i:i+12]))
        c_array += "    " + line + ",\n"
    c_array = c_array.rstrip(",\n") + "\n};\n\n"
    return c_array


image_folder = "./images"
output_file = "image_data.c"

with open(output_file, "w") as out:
    out.write("#include <stdint.h>\n\n")
    for i in range(12):
        img_name = os.path.join(image_folder, f"image_{i}.png")
        array_name = f"image_{i}"
        print(f"Converting: {img_name}")
        c_array = convert_image_to_c(img_name, array_name)
        out.write(c_array)

print(f"All images converted! Output saved to {output_file}")
