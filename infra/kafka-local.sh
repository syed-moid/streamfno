#!/usr/bin/env bash
#
# kafka-local.sh — folder-local Kafka lab, no Docker, no admin rights, no system install.
#
# Everything (JDK, Kafka, data, logs, configs) lives under one directory.
# Delete the directory and the whole lab is gone. The uv model, applied to Kafka.
#
#   ./kafka-local.sh setup          download JDK (if needed) + Kafka into ./kafka-lab
#   ./kafka-local.sh start          start an N-broker KRaft cluster on localhost
#   ./kafka-local.sh status         show which nodes are up
#   ./kafka-local.sh smoke          create a topic, produce & consume a message
#   ./kafka-local.sh stop           stop all nodes
#   ./kafka-local.sh clean          stop + wipe data (keeps downloads)
#
# Tunables (env vars):
#   LAB_DIR        install root            (default: ./kafka-lab)
#   KAFKA_VERSION  Kafka version           (default: 4.0.0 — bump freely)
#   BROKERS        number of nodes         (default: 3, KRaft combined mode)
#   BASE_PORT      first client port       (default: 9092; node i uses BASE_PORT+100*i)
#
# Notes:
#   * KRaft mode — no ZooKeeper.
#   * Uses your existing `java` if it's JDK 17+; otherwise downloads Temurin 21
#     into the lab folder (macOS arm64/x64 and Linux supported).
#   * All state in $LAB_DIR/data — safe to wipe between experiments.

set -euo pipefail

LAB_DIR="${LAB_DIR:-$(pwd)/kafka-lab}"
KAFKA_VERSION="${KAFKA_VERSION:-4.0.0}"
SCALA_VERSION="2.13"
BROKERS="${BROKERS:-3}"
BASE_PORT="${BASE_PORT:-9092}"

KAFKA_DIST="kafka_${SCALA_VERSION}-${KAFKA_VERSION}"
KAFKA_HOME="$LAB_DIR/$KAFKA_DIST"
DATA_DIR="$LAB_DIR/data"
CONF_DIR="$LAB_DIR/conf"
LOG_DIR="$LAB_DIR/logs"
PID_DIR="$LAB_DIR/pids"

# ---------------------------------------------------------------- helpers ----

msg()  { printf '\033[1;34m[kafka-local]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[kafka-local] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

client_port()     { echo $(( BASE_PORT + 100 * ($1 - 1) ));  }   # 9092, 9192, 9292...
controller_port() { echo $(( BASE_PORT + 100 * ($1 - 1) + 1 )); } # 9093, 9193, 9293...

detect_platform() {
  local os arch
  case "$(uname -s)" in
    Darwin) os="mac" ;;
    Linux)  os="linux" ;;
    *) die "unsupported OS: $(uname -s)" ;;
  esac
  case "$(uname -m)" in
    arm64|aarch64) arch="aarch64" ;;
    x86_64)        arch="x64" ;;
    *) die "unsupported arch: $(uname -m)" ;;
  esac
  echo "$os $arch"
}

find_java() {
  # Prefer a lab-local JDK, then the system java if it's modern enough.
  local lab_jdk
  # depth 6: macOS Temurin tarballs nest as jdk/<release>/Contents/Home/bin/java
  lab_jdk=$(find "$LAB_DIR" -maxdepth 6 -name java -path '*/bin/java' 2>/dev/null | head -1 || true)
  if [[ -n "$lab_jdk" ]]; then echo "$lab_jdk"; return; fi
  if command -v java >/dev/null 2>&1; then
    local ver
    ver=$(java -version 2>&1 | grep -oE 'version "([0-9]+)' | grep -oE '[0-9]+' | head -1)
    if [[ -n "${ver:-}" && "$ver" -ge 17 ]]; then command -v java; return; fi
  fi
  echo ""
}

ensure_java() {
  local java_bin
  java_bin=$(find_java)
  if [[ -n "$java_bin" ]]; then msg "java: $java_bin"; return; fi

  msg "no JDK 17+ found — downloading Temurin 21 into the lab folder..."
  read -r os arch <<< "$(detect_platform)"
  local url="https://api.adoptium.net/v3/binary/latest/21/ga/${os}/${arch}/jdk/hotspot/normal/eclipse"
  mkdir -p "$LAB_DIR/jdk"
  curl -fL --retry 3 -o "$LAB_DIR/jdk.tar.gz" "$url" \
    || die "JDK download failed. If your proxy blocks api.adoptium.net, download any JDK 17+ tarball manually and extract it under $LAB_DIR/jdk/"
  tar -xzf "$LAB_DIR/jdk.tar.gz" -C "$LAB_DIR/jdk" && rm -f "$LAB_DIR/jdk.tar.gz"
  java_bin=$(find_java)
  [[ -n "$java_bin" ]] || die "JDK extracted but java binary not found under $LAB_DIR/jdk"
  msg "java installed: $java_bin"
}

export_java_env() {
  local java_bin
  java_bin=$(find_java)
  [[ -n "$java_bin" ]] || die "no JDK — run: $0 setup"
  export JAVA_HOME="${java_bin%/bin/java}"
  export PATH="$JAVA_HOME/bin:$PATH"
}

# ----------------------------------------------------------------- setup -----

do_setup() {
  mkdir -p "$LAB_DIR" "$DATA_DIR" "$CONF_DIR" "$LOG_DIR" "$PID_DIR"
  ensure_java

  if [[ -x "$KAFKA_HOME/bin/kafka-storage.sh" ]]; then
    msg "Kafka $KAFKA_VERSION already present."
  else
    msg "downloading Kafka $KAFKA_VERSION..."
    local primary="https://downloads.apache.org/kafka/${KAFKA_VERSION}/${KAFKA_DIST}.tgz"
    local archive="https://archive.apache.org/dist/kafka/${KAFKA_VERSION}/${KAFKA_DIST}.tgz"
    curl -fL --retry 3 -o "$LAB_DIR/kafka.tgz" "$primary" \
      || curl -fL --retry 3 -o "$LAB_DIR/kafka.tgz" "$archive" \
      || die "Kafka download failed from both mirrors. Fetch ${KAFKA_DIST}.tgz manually and place it at $LAB_DIR/kafka.tgz, then re-run setup."
    tar -xzf "$LAB_DIR/kafka.tgz" -C "$LAB_DIR" && rm -f "$LAB_DIR/kafka.tgz"
    msg "Kafka extracted to $KAFKA_HOME"
  fi

  generate_configs
  format_storage
  msg "setup complete. next: $0 start"
}

generate_configs() {
  msg "generating $BROKERS KRaft combined-mode node configs..."
  local voters="" i
  for i in $(seq 1 "$BROKERS"); do
    voters+="${voters:+,}${i}@localhost:$(controller_port "$i")"
  done

  for i in $(seq 1 "$BROKERS"); do
    local cp kp rf
    cp=$(client_port "$i"); kp=$(controller_port "$i")
    rf=$(( BROKERS < 3 ? BROKERS : 3 ))
    mkdir -p "$DATA_DIR/node-$i"
    cat > "$CONF_DIR/node-$i.properties" <<EOF
# written by kafka-local.sh — node $i of $BROKERS
process.roles=broker,controller
node.id=$i
controller.quorum.voters=$voters
listeners=PLAINTEXT://localhost:$cp,CONTROLLER://localhost:$kp
advertised.listeners=PLAINTEXT://localhost:$cp
inter.broker.listener.name=PLAINTEXT
controller.listener.names=CONTROLLER
listener.security.protocol.map=PLAINTEXT:PLAINTEXT,CONTROLLER:PLAINTEXT
log.dirs=$DATA_DIR/node-$i
num.partitions=8
offsets.topic.replication.factor=$rf
transaction.state.log.replication.factor=$rf
transaction.state.log.min.isr=$(( rf > 1 ? rf - 1 : 1 ))
# --- telemetry-friendly settings for the StreamFNO lab ---
# fine-grained metrics recording; scrape via JMX on port $(( kp + 1 ))
metrics.recording.level=DEBUG
# fast retention enforcement so minutes-scale retention.ms (e11) expires
# on schedule instead of at the 5-minute default check cadence
log.retention.check.interval.ms=10000
EOF
  done
  msg "configs written to $CONF_DIR"
}

format_storage() {
  export_java_env
  local id_file="$LAB_DIR/cluster.id"
  if [[ ! -f "$id_file" ]]; then
    "$KAFKA_HOME/bin/kafka-storage.sh" random-uuid > "$id_file"
  fi
  local cid; cid=$(cat "$id_file")
  local i
  for i in $(seq 1 "$BROKERS"); do
    if [[ ! -f "$DATA_DIR/node-$i/meta.properties" ]]; then
      "$KAFKA_HOME/bin/kafka-storage.sh" format -t "$cid" -c "$CONF_DIR/node-$i.properties" >/dev/null
      msg "formatted storage for node $i (cluster $cid)"
    fi
  done
}

# ------------------------------------------------------------- lifecycle -----

do_start() {
  export_java_env
  [[ -x "$KAFKA_HOME/bin/kafka-server-start.sh" ]] || die "not set up — run: $0 setup"
  local i
  for i in $(seq 1 "$BROKERS"); do
    if is_up "$i"; then msg "node $i already running"; continue; fi
    local jmx_port=$(( $(controller_port "$i") + 1 ))
    KAFKA_HEAP_OPTS="-Xms256m -Xmx512m" \
    JMX_PORT="$jmx_port" \
    KAFKA_JMX_OPTS="-Dcom.sun.management.jmxremote -Dcom.sun.management.jmxremote.authenticate=false -Dcom.sun.management.jmxremote.ssl=false -Dcom.sun.management.jmxremote.port=$jmx_port -Djava.rmi.server.hostname=localhost" \
    LOG_DIR="$LOG_DIR/node-$i" \
      nohup "$KAFKA_HOME/bin/kafka-server-start.sh" "$CONF_DIR/node-$i.properties" \
      > "$LOG_DIR/node-$i.out" 2>&1 &
    echo $! > "$PID_DIR/node-$i.pid"
    msg "node $i starting (pid $!, client :$(client_port "$i"), jmx :$jmx_port)"
  done
  msg "waiting for cluster..."
  sleep 5
  do_status
  msg "bootstrap servers: $(bootstrap_list)"
}

bootstrap_list() {
  local i out=""
  for i in $(seq 1 "$BROKERS"); do out+="${out:+,}localhost:$(client_port "$i")"; done
  echo "$out"
}

is_up() {
  local pid_file="$PID_DIR/node-$1.pid"
  [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null
}

do_status() {
  local i
  for i in $(seq 1 "$BROKERS"); do
    if is_up "$i"; then
      msg "node $i: UP   (pid $(cat "$PID_DIR/node-$i.pid"), client :$(client_port "$i"))"
    else
      msg "node $i: DOWN"
    fi
  done
}

do_smoke() {
  export_java_env
  local bs; bs=$(bootstrap_list)
  local topic="streamfno-smoke"
  msg "creating topic '$topic'..."
  "$KAFKA_HOME/bin/kafka-topics.sh" --bootstrap-server "$bs" --create \
    --topic "$topic" --partitions 8 --replication-factor "$(( BROKERS < 3 ? BROKERS : 3 ))" \
    2>/dev/null || msg "(topic may already exist)"
  msg "producing one message..."
  echo "hello from kafka-local $(date)" | \
    "$KAFKA_HOME/bin/kafka-console-producer.sh" --bootstrap-server "$bs" --topic "$topic" >/dev/null
  msg "consuming it back..."
  "$KAFKA_HOME/bin/kafka-console-consumer.sh" --bootstrap-server "$bs" --topic "$topic" \
    --from-beginning --max-messages 1 --timeout-ms 15000
  msg "smoke test passed ✔  cluster is functional."
  msg "describe: $KAFKA_HOME/bin/kafka-topics.sh --bootstrap-server $bs --describe --topic $topic"
}

do_stop() {
  local i
  for i in $(seq 1 "$BROKERS"); do
    if is_up "$i"; then
      kill "$(cat "$PID_DIR/node-$i.pid")" && msg "node $i stopped"
      rm -f "$PID_DIR/node-$i.pid"
    fi
  done
}

do_clean() {
  do_stop
  sleep 2
  rm -rf "$DATA_DIR" "$LOG_DIR" "$PID_DIR" "$LAB_DIR/cluster.id"
  msg "data wiped. downloads kept. re-run: $0 setup"
}

# -------------------------------------------------------------------- cli ----

case "${1:-}" in
  setup)  do_setup ;;
  start)  do_start ;;
  stop)   do_stop ;;
  status) do_status ;;
  smoke)  do_smoke ;;
  clean)  do_clean ;;
  *) grep '^#' "$0" | head -30 | sed 's/^# \{0,1\}//'; exit 1 ;;
esac
