# Python and Java versions are chosen for the PySpark 3.5.5 course environment.
FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    JAVA_HOME=/opt/java/openjdk \
    SPARK_LOCAL_IP=127.0.0.1 \
    PYSPARK_PYTHON=/usr/local/bin/python

# Resolve the Java directory for the container's architecture (arm64 or amd64).
RUN apt-get update \
    && apt-get install -y --no-install-recommends openjdk-17-jre-headless procps bash \
    && mkdir -p /opt/java \
    && ln -s "/usr/lib/jvm/java-17-openjdk-$(dpkg --print-architecture)" "$JAVA_HOME" \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="${JAVA_HOME}/bin:${PATH}"

WORKDIR /app
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt \
    && python -m pip check \
    && python -m ipykernel install --sys-prefix --name assignment1 \
       --display-name "Python (Assignment 1 Docker)"

EXPOSE 8888

# Keep Jupyter's default token authentication. The URL is printed at startup.
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--ServerApp.port_retries=0", "--no-browser", "--allow-root", "--ServerApp.root_dir=/app"]
