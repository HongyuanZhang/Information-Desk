# InfoDesk

An open-domain QA machine.

## How to use our QA machine?

1. Run `mkdir logs`. Download the checkpoints (the "ckpt" files) and the Stack Exchange data sets (`output.csv` and `output_f_a.csv`) from https://drive.google.com/open?id=1h8ZgCx30YiFdUdFQsxcDJlFF0HfRVFm7. Put these files into the `logs` folder that you just made.

2. Edit the question string in `infodesk.py`

3. Run `python3 infodesk.py`. Make sure you are using python3 and have the required libraries, i.e. pytorch, numpy, bs4, nltk etc. installed on your system. 

## Files

*infodesk.py*: the main prediction code for users to use.

*kse_predict.py*: the prediction code for key sentence extractor.

*kse_train.py*: training code for key sentence extractor.

*kse_test.py*: evaluation code for key sentence extractor.

*lstm.py*: models used in key sentence extractor.

*gan_predict.py*: the prediction code for answer generator.

*gan_train.py*: training code for answer generator.

*gan_discriminator.py*: model used as discriminator in answer generator.

*gan_generator.py*: model used as generator in answer generator.

*dataloader.py*: important data loading tools, including webpage loader and Google NQ bucketing function.

*stack_overflow_crawler.py*: the crawler for extracting dataset from Stack Exchange.

*utility.py*: utilities for training models.
