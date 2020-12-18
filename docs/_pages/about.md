---
layout: page
title: About
permalink: /about/
---

The goal of this project is to build a camera that can track airplanes as they fly by and take pictures. It will collect ADS-B messages to find out where nearby planes are, do some math to figure out what direction to look, point a camera at the plane and take a picture. We will be trying a couple of different approaches for moving the camera around, including:
- [Axis m5525-e PTZ camera](https://www.axis.com/en-us/products/axis-m5525-e): This is a bit of a higher end option, but it combines precise control, quality optics and an easy API control.
- [Pimoroni Pan Tilt hat](https://shop.pimoroni.com/products/pan-tilt-hat?variant=22408353287): A much lower cost option, but only 180 degree panning is supported and there is no zoom capability.
- [DJI Ronin](https://www.dji.com/ronin-sc?site=brandsite&from=nav) Camera Gimbal: Great at stabilizing a camera, and it can accept movement commands over serial.

## Design
The different components for the systems will be modular, allowing for the different camera systems to be swapped out. The components will communicate with each other using MQTT messages. The initial system will assume it is in a fixed location, but a future version could support being moved and reconfigure themselves using GPS and a compass heading.

## Build Journal
We will be posting updates every few days to update on progress, design decisions, and lessons learned during the buildout.

## Inspiration
We are not the first to think of this concept, we first saw it [here](http://simonaubury.com/the-pi-plane-project-whole-write-up/)