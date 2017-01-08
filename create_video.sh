ffmpeg -framerate 25 -f image2 -pattern_type glob -i '*.png' -b:v 5M -pix_fmt yuv420p out.avi
