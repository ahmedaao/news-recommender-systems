# Use of the Python 3.10.12 base image
FROM python:3.10.12

# Definition of the working directory in the container
WORKDIR /repo

# Copy the necessary files into the container
COPY app/frontend/streamlit_app.py /repo/
COPY requirements.txt /repo/
COPY src/ /repo/src/
COPY app/backend/dataset.pickle /repo/app/backend/
COPY setup.py /repo/

# Installation of Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Definition of the ROOT_DIR environment variable
ENV ROOT_DIR=/repo/

# Exposure of the port on which the application will listen
EXPOSE 8501

# Command to execute the Streamlit application
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
