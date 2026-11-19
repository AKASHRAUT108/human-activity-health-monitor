# KTH Dataset Preparation

## Purpose

The KTH dataset contains six human action categories:

- Boxing
- Handclapping
- Handwaving
- Jogging
- Running
- Walking

The original videos contain repeated action sequences. This pipeline uses
the official `00sequences.txt` annotations to create one manifest record
per annotated action repetition.

An entire source video must not be treated as one action sample.

## Dataset layout

Expected input:

```text
data/
  raw/
    kth/
      downloads/
        boxing.zip
        handclapping.zip
        handwaving.zip
        jogging.zip
        running.zip
        walking.zip
      annotations/
        00sequences.txt