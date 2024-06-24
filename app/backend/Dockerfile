# Use of the Python 3.10.12 base image
FROM python:3.10.12

# Definition of the working directory in the container
WORKDIR /repo

# Copy the necessary files into the container
COPY app/backend/fastapi_app.py /repo/
COPY requirements.txt /repo/
COPY app/backend/dataset.pickle /repo/app/backend/
COPY src/ /repo/src/
COPY models/articles_embeddings.pickle /repo/models/
COPY models/model_based_svd.pickle /repo/models/
COPY setup.py /repo/

# Installation of Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Definition of the ROOT_DIR environment variable
ENV ROOT_DIR=/repo/

# Exposure of the port on which the application will listen
EXPOSE 8000

# Command to execute the FastAPI application
CMD ["uvicorn", "fastapi_app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
