#!/bin/bash

# RefImage Backend Server Start Script with Detailed Logging
# Created: 2025-07-27
# Purpose: Start backend server with comprehensive logging for debugging

set -e  # Exit on any error

# Configuration
BACKEND_DIR="/home/mako10k/imagestore/backend"
LOG_DIR="${BACKEND_DIR}/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Log files
STDOUT_LOG="${LOG_DIR}/refimage_stdout_${TIMESTAMP}.log"
STDERR_LOG="${LOG_DIR}/refimage_stderr_${TIMESTAMP}.log"
COMBINED_LOG="${LOG_DIR}/refimage_combined_${TIMESTAMP}.log"

# Create log directory if it doesn't exist
mkdir -p "${LOG_DIR}"

echo "================================================"
echo "RefImage Backend Server Startup"
echo "Timestamp: $(date)"
echo "Log Directory: ${LOG_DIR}"
echo "STDOUT Log: ${STDOUT_LOG}"
echo "STDERR Log: ${STDERR_LOG}"
echo "Combined Log: ${COMBINED_LOG}"
echo "================================================"

# Change to backend directory
cd "${BACKEND_DIR}"

# Show current environment
echo "Current working directory: $(pwd)" | tee -a "${COMBINED_LOG}"
echo "Python version: $(python --version)" | tee -a "${COMBINED_LOG}"
echo "Python executable: $(which python)" | tee -a "${COMBINED_LOG}"
echo "Environment variables:" | tee -a "${COMBINED_LOG}"
printenv | grep -E "(CUDA|PATH|PYTHON)" | tee -a "${COMBINED_LOG}"

echo "================================================" | tee -a "${COMBINED_LOG}"
echo "Starting RefImage server with uvicorn..." | tee -a "${COMBINED_LOG}"
echo "Command: python -m uvicorn refimage.main:app --host 0.0.0.0 --port 8000" | tee -a "${COMBINED_LOG}"
echo "Start time: $(date)" | tee -a "${COMBINED_LOG}"
echo "================================================" | tee -a "${COMBINED_LOG}"

# Start server with detailed logging
# stdout to both file and terminal, stderr to separate file and combined
python -m uvicorn refimage.main:app --host 0.0.0.0 --port 8000 \
    > >(tee "${STDOUT_LOG}" | tee -a "${COMBINED_LOG}") \
    2> >(tee "${STDERR_LOG}" | tee -a "${COMBINED_LOG}" >&2)

# This will only execute if the server exits
EXIT_CODE=$?
echo "================================================" | tee -a "${COMBINED_LOG}"
echo "Server stopped at: $(date)" | tee -a "${COMBINED_LOG}"
echo "Exit code: ${EXIT_CODE}" | tee -a "${COMBINED_LOG}"
echo "================================================" | tee -a "${COMBINED_LOG}"

exit ${EXIT_CODE}
