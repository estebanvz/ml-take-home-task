I will solve this problem in different phases:

# Phase 1: Research

The first step is to research potential solutions in the literature. I consulted the following articles:


* [Handwritten Optical Character Recognition (OCR): A Comprehensive Systematic Literature Review (SLR)](https://ieeexplore.ieee.org/abstract/document/9151144)

* [Optical Character Recognition (OCR) in Handwritten Characters Using Convolutional Neural Networks to Assist in Exam Reader System](https://ieeexplore.ieee.org/abstract/document/10551027)


These articles highlight that the main approaches for this type of problem include Convolutional Neural Networks (CNNs), Support Vector Machines, and Transformers.

# Phase 2: Development

In this phase, I set up the technologies needed for the project. To ensure reproducibility, I used Docker with Visual Studio Code Containers. The configuration file is the Dockerfile.

[Dockerfile](./Dockerfile)

# Phase 3: Basemodel

Based on the research, CNNs are recommended for this task. I started with the EasyOCR model, as it provides the necessary parameters for this task.

[EasyOCR](https://github.com/JaidedAI/EasyOCR)

# Phase 4: Model Selection

Since the problem involves handwriting, I found a model that was trained using the IAM dataset, as mentioned in the articles and the Microsoft documentation. This model is the TrOCR base-sized model, fine-tuned on IAM.

[IAM Handwriting Database](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database)

[TrOCR (base-sized model, fine-tuned on IAM)](https://huggingface.co/microsoft/trocr-base-handwritten)

# Phase 5: Model Management

I created a Python library called ocr_models.py to manage the models. Additionally, I implemented a wrapper so that each model can be used interchangeably.

[ocr_models.py](ocr_models.py)

# Phase 6: Image Transformation

Since the results were varied, I modified the Microsoft model by adding image transformations, as some images were old or had low contrast. Thanks to the wrapper, this modification was straightforward.

[main.py](main.py)

# Phase 7: Model Comparison and Next Steps

In the main script I compare the models and select the model with higher confidence. Also, I extract the configuration of the image transformation as configuration variable so It can be used to optimize the models in the future using Optimization algorithms in the hyper parameters.

# Phase 8: CSV Output

I wrote the results to a CSV file, including all required fields, with an additional field specifying the model used. This will help track which configuration and model were used.

---

# Storied Machine Learning Take-Home Task
Take home task for machine learning candidates

In this repository is a zip file containing 100 image snippets with handwriting. Using a handwriting recognition model of your choice, or one you fine-tune, please execute handwriting recognition for these snippets.

Please use open-source or non-paid models, rather than subscription-based solutions like OpenAI/Anthropic/Gemini.

The results should be in a CSV file with the following columns:

- snippet_name
- label (the recognized text)
- confidence_score (a value between 0 and 1, like 0.78)
- If any snippet is blank, it’s fine to leave the label and confidence_score fields empty.

Please commit all code to a public GitHub repository that you can share with us for review.

We know this is a very small amount of data, and the goal is not so much to see how accurate the output is as it is to illustrate your process, how you evaluate and incorporate models, and how you write and structure your code.

Thanks for taking the time to do this, and please feel free to reach out if you have any questions.
