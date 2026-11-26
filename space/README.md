---
title: Lakeshore Ticket Sorter
emoji: 🚲
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
license: apache-2.0
---

# Lakeshore Ticket Sorter

Type a short support message and the app returns a score for each of the four queues:
billing, bike_fault, station and account. The model is DistilBERT fine-tuned on a few
hundred invented messages for Humber AIGC 5005 Lab 09, so treat the scores as
suggestions, not decisions.
