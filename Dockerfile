# syntax=docker/dockerfile:1

# Create container using 3.12 Python version
ARG PYTHON_VERSION=3.12
# Use slim Python version as it creates smaller image
FROM python:${PYTHON_VERSION}-slim

# Will not create __pycache__ and *.pyc
ENV PYTHONDONTWRITEBYTECODE=1

# Makes logs appear immediately
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy dependencies requirements
COPY requirements.txt .
# Install the dependencies,
# make sure to use --no-cache-dir to create a smaller image
RUN pip install --no-cache-dir -r requirements.txt

# Copy the source code into the container
COPY . .

# Expose the port that the streamlit application listens on
EXPOSE 8501

# Run the streamlit app
CMD ["streamlit", "run", "app/app.py", "--server.address=0.0.0.0"]