import os
from subprocess import run

pngs = []

for path, subdirs, files in os.walk('.'):
    some_pngs = [os.path.join(path, f) for f in files if os.path.splitext(f)[1] == '.png']
    pngs.extend(some_pngs)

for png_path in pngs:
    no_ext = os.path.splitext(png_path)[0]
    jpg_path = no_ext+'.jpg'
    left_jpg_path = no_ext+'_left.jpg'
    if 'image-samples/cifar10' in png_path:
        factor = 34
    elif 'image-samples/ffhq' in png_path:
        factor = 258
    else:
        continue
    cmd = f"convert {png_path} -chop {factor*2}x0 {jpg_path}"
    other_cmd = f"convert {png_path} -chop {factor*5}x0+{factor*2}+0 {left_jpg_path}"
    print(cmd)
    print(other_cmd)
