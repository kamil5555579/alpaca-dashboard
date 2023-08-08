FROM python:3.10
WORKDIR /app

# Set the DATABASE_HOST environment variable
ENV DATABASE_HOST=db

# Install system-wide dependencies (if any) using apt-get, etc.

# Change to the ALPACA program directory
WORKDIR /app/python-analyses

# Create a virtual environment for the ALPACA program
RUN python -m venv venv
ENV PATH="/app/python-analyses/venv/bin:$PATH"

# Copy the ALPACA program files into the container
COPY python-analyses/ /app/python-analyses/

# Install dependencies for the ALPACA program
COPY python-analyses/requirements.txt .
RUN venv/bin/pip install --no-cache-dir -r requirements.txt

# Set the PYTHONPATH to include the ALPACA module's root directory
ENV PYTHONPATH="/app/python-analyses:$PYTHONPATH"

# Set the working directory to the directory containing the script
WORKDIR /app/python-analyses/ALPACA/applications

# Initialize alpaca table in alpaca database
#RUN /app/python-analyses/venv/bin/python alpaca_to_database.py --first_run 387116 --last_run 387117

# Change to the app directory
WORKDIR /app/alpaca-app

# Create a virtual environment for the Dash app
RUN python -m venv venv
ENV PATH="/app/alpaca-app/venv/bin:$PATH"

# Copy your Dash app files into the container
COPY alpaca-app/ /app/alpaca-app/

# Install dependencies for your Dash app
COPY alpaca-app/requirements.txt .
RUN venv/bin/pip install --no-cache-dir -r requirements.txt

# Set the command to run Gunicorn with your app
CMD venv/bin/gunicorn --bind 0.0.0.0:5000 app:server --workers 4 --timeout 120