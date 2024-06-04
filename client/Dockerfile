# Using a slim version of Python 3.10
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application's code into the container
COPY . /app/

# Make port 8501 available to the world outside this container
EXPOSE 8501

# Run client.py when the container launches
CMD ["python", "client.py"]

