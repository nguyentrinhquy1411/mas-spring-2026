FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY pyproject.toml uv.lock ./
RUN pip install uv && uv pip install --system -e .

# Copy the rest of the application
COPY . .

# Expose port
EXPOSE 8000

# Run the FastAPI app
CMD ["python", "-m", "uvicorn", "house_price_prediction.webapp.main:app", "--host", "0.0.0.0", "--port", "8000"]
