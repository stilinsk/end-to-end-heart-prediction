# Use a specific Python base image
FROM python:3.7

# Set the working directory inside the container
WORKDIR /app

# Copy the project files to the working directory
COPY . .

# Install project dependencies
RUN pip install -r requirements.txt

# Expose the port for the application (if applicable)
EXPOSE 5000

# Set the entrypoint and default command
ENTRYPOINT ["python"]
CMD ["app.py", "-m", "flask", "--host=0.0.0.0"]