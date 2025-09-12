FROM runpod/pytorch:0.7.0-cu1281-torch271-ubuntu2204


WORKDIR /workspace

# Copy requirements first for better caching
COPY requirements.txt /workspace/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the training script - explicitly to /workspace/
COPY train.py /workspace/train.py

# Verify the file exists
RUN ls -la /workspace/train.py

# Set the default command to run the training script
CMD ["python", "/workspace/train.py"]