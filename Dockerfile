# syntax=docker/dockerfile:1
FROM python:3.12-slim
WORKDIR /app
# Optional: trust an extra root CA (antivirus / corporate HTTPS inspection), e.g.
#   docker build --secret id=extra_ca,src=my-root-ca.pem -t maintenance-copilot .
RUN --mount=type=secret,id=extra_ca,required=false \
    if [ -f /run/secrets/extra_ca ]; then \
      cp /run/secrets/extra_ca /usr/local/share/ca-certificates/extra_ca.crt && update-ca-certificates; \
    fi
ENV PIP_CERT=/etc/ssl/certs/ca-certificates.crt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python generate_data.py
EXPOSE 8501
# Pass secrets at runtime:  docker run -p 8501:8501 --env-file .env maintenance-copilot
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", "--browser.gatherUsageStats=false"]
