FROM python:3.10
WORKDIR /app

# Install system-wide dependencies (if any) using apt-get, etc.

# Create a virtual environment for the Dash app
RUN python -m venv /venv/dash
ENV PATH="/venv/dash/bin:$PATH"

# Copy your Dash app files into the container
COPY alpaca-app/ /app/alpaca-app/

# Install dependencies for your Dash app
COPY alpaca-app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# Create a virtual environment for the ALPACA program
RUN python -m venv /venv/alpaca
ENV PATH="/venv/alpaca/bin:$PATH"

# Copy the ALPACA program files into the container
COPY python-analyses/ /app/python-analyses/

# Install dependencies for the ALPACA program
COPY python-analyses/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV PATH="/venv/dash/bin:$PATH"
WORKDIR /alpaca-app

CMD /venv/dash/bin/gunicorn --bind 0.0.0.0:5000 app:server --workers 4 --timeout 120