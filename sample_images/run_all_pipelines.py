import subprocess

pipelines = ['gfap', 'hoechst', 'neun', 'sox9']
image_dir = 'sample_images'

for pipeline in pipelines:
    print(f"\nRunning {pipeline} on all images...\n")
    cmd = [pipeline] + [f"{image_dir}/" + img for img in [
        "camera.png", "cells.png", "checker.png", "clock.png", "coins.png",
        "horse.png", "moon.png", "page.png", "rocket.png", "text.png"
    ]]
    subprocess.run(cmd)
