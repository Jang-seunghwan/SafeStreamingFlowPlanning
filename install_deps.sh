#!/usr/bin/env bash
# System dependencies (Ubuntu 22.04): ROS 2 Humble packages for Gazebo Fortress
# (ros_gz), Nav2 and xacro, plus colcon. Assumes ROS 2 Humble is already
# installed under /opt/ros/humble. Python packages: see requirements.txt.
# Run with: bash install_deps.sh

set -e

echo "[1/3] apt update"
sudo apt update

echo "[2/3] Installing ROS 2 / Gazebo / Nav2 / tooling packages"
sudo apt install -y \
  ros-humble-ros-gz \
  ros-humble-ros-gz-bridge \
  ros-humble-ros-gz-sim \
  ros-humble-ros-gz-interfaces \
  ros-humble-navigation2 \
  ros-humble-nav2-bringup \
  ros-humble-nav2-simple-commander \
  ros-humble-xacro \
  python3-colcon-common-extensions \
  python3-rosdep \
  python3-vcstool \
  python3-pip

echo "[3/3] rosdep init (skipped if already initialized)"
if [ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]; then
  sudo rosdep init || true
fi
rosdep update
