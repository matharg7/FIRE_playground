# 1. Define the Base Environment
FROM nvcr.io/nvidia/pytorch:24.07-py3

# 2. Install Custom Python Dependencies
# Copying just the requirements file first allows Docker to cache this layer.
# If you change your code but not your requirements, it won't have to reinstall everything!
COPY vision/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# 3. Set the Working Directory
# This is the equivalent of running 'cd /workspace' inside the container.
WORKDIR /workspace

# 4. (Optional) Copy Your Project Code
# If you want to bake your code directly into the image for ultimate reproducibility:
# COPY . /workspace
