# Report: is fine-tuning worth it for sorting support messages?

<!--
Write this yourself AFTER you run the experiments, using the numbers in
reports/summary.csv and reports/final.json. Aim for 300-450 words, written for the
support lead (not for the instructor). Delete these comments before you submit.
-->

## The result in one sentence

<!-- e.g. "At 400 training messages the best baseline (TF-IDF) averaged X on
validation (range A-B) and the best fine-tuned model (full / head only) averaged Y
(range C-D); on the test rows they scored X' and Y'." -->

## Where fine-tuning helped and where it did not

<!-- Walk along the learning curve: 50, 100, 200, 400. At each size, do the ranges
overlap? If they overlap, say you cannot call one better. If fine-tuning never beat
TF-IDF, say so plainly. -->

## Why I think so

<!-- Name at least two reasons and say which you CHECKED and HOW. Useful evidence in
this repo:
- reports/final.json -> "validation_vague_vs_clear": accuracy on the 28 vague
  validation messages vs the 212 clear ones. If every method is near chance on the
  vague ones and high on the clear ones, the ceiling comes from the labels, not
  the model.
- The generator repeats ~13 sentence patterns per queue, so word counts (TF-IDF)
  already capture them: compare how close TF-IDF gets to the ceiling.
- reports/tuning.csv: did head-only need a bigger learning rate than full?
- Seed noise: how wide are the ranges at 50 rows vs 400 rows?
- reports/training_log.csv: does the training loss keep falling while validation
  accuracy stops improving? -->

## What my own 40 messages showed

<!-- Compare the own-message scores with the test scores. A big drop means the
generated data flatters the models. Which method dropped least? Why might a
pretrained model cope better (or not) with new wording and spelling slips? -->

## Recommendation

<!-- One recommendation that follows from the evidence, and one thing that would
change it (e.g. a few thousand real labelled messages, or real messages with
wording the generator never uses). -->

## What this evidence cannot tell you

<!-- An honest limit, e.g. the text is invented, only 3 seeds, settings tuned at one
size, no real customers, no cost or speed measured on a server. -->
