---
layout: post
title: Hello World!
categories: [Progress]
---

Welcome! We are just getting things started. Our first order of business will be to collecting plane positions using ADS-B and a Software Defined Radio.

## Background
 [RTL-SDRs](https://osmocom.org/projects/rtl-sdr/wiki/Rtl-sdr) are a class of popular, low-cost SDR that are based on a repurposed digital TV chip. They can only cover about 2.4MHz of spectrum at a time, but that is more than enough for our purposes. [Dump1090](https://github.com/antirez/dump1090) is a popular program that uses an RTL-SDR to capture and decode ADS-B messages. These decoded messages are made available over a network port and there is both a text and web user interface for monitoring planes. 

## First Task
Our first task is going to be placing dump1090 in a docker container and then creating a small program that can convert the network messages from dump1090 into MQTT messages. We will report back here on our progress!
