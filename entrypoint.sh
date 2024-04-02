#!/bin/bash
# This script just waits indefinitely, allowing users to manually execute commands inside the container
echo "Container is ready."
exec "$@"