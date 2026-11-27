# Report: is fine-tuning worth it for sorting support messages?


## The result in one sentence

At 400 training messages, full fine-tuning scored 0.911 on validation, which was better than TF-IDF, which scored 0.858.

## Where fine-tuning helped and where it did not

With about 50 messages, there was no clear winner because the scores varied a lot across seeds, and full fine-tuning ranged from 0.50 to 0.75. With about 200 messages, full fine-tuning was clearly better, as its worst score of 0.883 was still higher than TF-IDF's best score of 0.846. Head-only training stopped improving at around 0.83, which was lower than TF-IDF.

## Why I think so

One reason is that some messages are too vague to understand, such as "This happened again today." I checked final.json and found that on the 28 vague validation messages, all methods scored around 0.25 to 0.32, while full fine-tuning scored 0.99 on the 212 clear messages.

Another reason is that head-only training cannot change the pretrained part of DistilBERT, so the model cannot fully adapt to this task. I checked the learning curve and saw that head-only training stayed around 0.83, while full fine-tuning continued improving and reached 0.91.

## What my own 40 messages showed

On my own 40 messages, TF-IDF got 0.975 (39 out of 40 correct), while full fine-tuning got 0.95 (38 out of 40 correct). The difference is only one message, so neither model was clearly better here, unlike the test results where full fine-tuning performed better. Both scored high because my messages clearly described one topic and did not include vague messages like the generated data.

## Recommendation

I would recommend using full fine-tuning when there are around 200 or more labelled messages because it performed better than TF-IDF, while TF-IDF is simpler and cheaper when there are fewer messages. I would change my recommendation if testing on real customer messages showed that full fine-tuning was not better, because TF-IDF might be enough in that case.

## What this evidence cannot tell you

The data was invented using a script that repeats similar sentence patterns, so the results may be lower on real customer messages. I only tested three seeds, which gives an idea of the variation but is not enough to know the exact range of performance. The model can also be too confident, as even "hi, good morning" was predicted as bike_fault with 76%, so a person should still check the predictions.
