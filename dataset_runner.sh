#!/usr/bin/env bash

SCRIPT="tts_generator.py"
ARGS="--links links.txt"
LOG="tts.log"
PIDFILE="tts.pid"

start() {
    if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
        echo "❌ Already running with PID $(cat $PIDFILE)"
        exit 1
    fi

    echo "🚀 Starting $SCRIPT..."
    nohup env PYTHONUNBUFFERED=1 \
        python3 -u "$SCRIPT" $ARGS \
        </dev/null >"$LOG" 2>&1 &
    echo $! >"$PIDFILE"
    echo "✅ Started with PID $(cat $PIDFILE). Logs: $LOG"
}

stop() {
    if [[ -f "$PIDFILE" ]]; then
        PID=$(cat "$PIDFILE")
        if kill -0 "$PID" 2>/dev/null; then
            echo "🛑 Stopping process $PID..."
            kill "$PID"
            rm -f "$PIDFILE"
            echo "✅ Stopped."
        else
            echo "⚠️ No process found for PID $PID. Cleaning up PID file."
            rm -f "$PIDFILE"
        fi
    else
        echo "⚠️ No PID file found. Is it running?"
    fi
}

status() {
    if [[ -f "$PIDFILE" ]]; then
        PID=$(cat "$PIDFILE")
        if kill -0 "$PID" 2>/dev/null; then
            echo "ℹ️ Running with PID $PID (log: $LOG)"
            exit 0
        else
            echo "⚠️ PID file exists but process not running."
            exit 1
        fi
    else
        echo "❌ Not running."
        exit 1
    fi
}

case "$1" in
    start) start ;;
    stop) stop ;;
    status) status ;;
    restart) stop; sleep 1; start ;;
    *)
        echo "Usage: $0 {start|stop|status|restart}"
        exit 1
        ;;
esac
