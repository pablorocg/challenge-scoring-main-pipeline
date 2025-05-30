# FOMO25 NGC Competition Server

Official competition server for processing FOMO25 challenge submissions.

## Architecture

- Receives containers via SFTP
- Validates and runs containers in secure environment
- Computes metrics against private ground truth data
- Sends results back via SFTP

## Setup

1. Run as root: `bash setup_environment.sh`
2. Place private data in `/data/fomo25/private_data/`
3. Configure SFTP keys in `/data/fomo25/config/`
4. Start services:
   ```bash
   systemctl start ngc-server
   systemctl start sftp-receiver
   ```

## Security Features

- Containers run with `--contain` and `--network none`
- Read-only access to test data
- No home directory access
- Runs as unprivileged user (uid 65534)
- 1-hour timeout per evaluation

## Directory Structure

- `incoming/`: New submissions from SFTP
- `processing/`: Currently being evaluated
- `processed/`: Completed evaluations
- `private_data/`: Ground truth data (by task type)
- `results/`: Computed metrics
- `logs/`: System and evaluation logs

## Monitoring

Run `./scripts/monitor_system.sh` to view real-time status.
