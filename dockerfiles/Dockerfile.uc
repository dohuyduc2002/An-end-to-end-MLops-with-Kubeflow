FROM unitycatalog/unitycatalog:v0.3.0

USER root

WORKDIR /home/unitycatalog

# Cài Spark 4.0.0
RUN wget https://archive.apache.org/dist/spark/spark-4.0.0/spark-4.0.0-bin-hadoop3.tgz && \
    mkdir -p /home/unitycatalog/bin/spark && \
    tar -xzf spark-4.0.0-bin-hadoop3.tgz -C /home/unitycatalog/bin/spark --strip-components=1 && \
    rm spark-4.0.0-bin-hadoop3.tgz

# Thêm jars cần thiết
WORKDIR /home/unitycatalog/bin/spark/jars
RUN wget -O gcs-connector-hadoop3-2.2.5-shaded.jar \
      https://repo1.maven.org/maven2/com/google/cloud/bigdataoss/gcs-connector/hadoop3-2.2.5/gcs-connector-hadoop3-2.2.5-shaded.jar && \
    wget -O delta-spark_2.13-4.0.0.jar \
      https://repo1.maven.org/maven2/io/delta/delta-spark_2.13/4.0.0/delta-spark_2.13-4.0.0.jar && \
    wget -O delta-storage-4.0.0.jar \
      https://repo1.maven.org/maven2/io/delta/delta-storage/4.0.0/delta-storage-4.0.0.jar && \
    wget -O unitycatalog-spark_2.13-0.3.0.jar \
      https://repo1.maven.org/maven2/io/unitycatalog/unitycatalog-spark_2.13/0.3.0/unitycatalog-spark_2.13-0.3.0.jar

RUN wget -O unitycatalog-client-0.3.0.jar \
  https://repo1.maven.org/maven2/io/unitycatalog/unitycatalog-client/0.3.0/unitycatalog-client-0.3.0.jar

RUN wget -O unitycatalog-server-0.3.0.jar \
  https://repo1.maven.org/maven2/io/unitycatalog/unitycatalog-server/0.3.0/unitycatalog-server-0.3.0.jar

# Trả về user gốc của UC server
USER unitycatalog

WORKDIR /home/unitycatalog

ENV PATH="$PATH:/home/unitycatalog/bin/spark/bin"
