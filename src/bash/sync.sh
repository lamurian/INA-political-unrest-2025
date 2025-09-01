#!/bin/bash

echo "Storing all files to the cloud"
rclone copy -PuL . "konsulin:/Projects/Research/Indonesia Political Unrest 2025"
echo "Finished!"
