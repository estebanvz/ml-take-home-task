# %%
import pandas as pd
import ocr_models

# Configuration for the models
configurations = [
    {
        "contrast_factor": 2.0,
        "sharpen_factor": 2.0,
        "threshold_factor": 0.8,
        "noise_reduction_size": 3,
        "sharpness_enhancement": True,
        "contrast_enhancement": True,
        "threshold_enhancement": True,
    },
    {
        "contrast_factor": 3.0,
        "sharpen_factor": 2.0,
        "threshold_factor": 0.8,
        "noise_reduction_size": 3,
        "sharpness_enhancement": False,
        "contrast_enhancement": True,
        "threshold_enhancement": False,
    },
]

# Models to be tested
models_testing = [
    ("EasyOCR", ocr_models.EasyOCR, {}),
    ("TrOCR", ocr_models.TrOCR, {}),
    ("TrOCRModified", ocr_models.TrOCRModified, configurations[0]),
    ("TrOCRModified2", ocr_models.TrOCRModified, configurations[1]),
]

# Store predictions from each model
predictions = {}
for model_name, model_class, config in models_testing:
    model_instance = model_class(name=model_name, configuration=config)
    model_instance.load_images("./snippets")
    results = model_instance.predict_images()
    predictions[model_name] = {"model": model_instance, "results": results}

# Gather highest confidence predictions
highest_confidence_predictions = []
for model_name, model_data in predictions.items():
    for image_path, label, confidence in model_data["results"]:
        # If this is the first time we encounter the image_path, or if we have a higher confidence
        existing_prediction = next(
            (item for item in highest_confidence_predictions if item[0] == image_path),
            None,
        )

        if not existing_prediction:
            # Add a new prediction with the highest confidence
            highest_confidence_predictions.append(
                [image_path, label, confidence, model_name]
            )
        else:
            # Update if current confidence is higher
            if confidence > existing_prediction[2]:
                existing_prediction[1:] = [label, confidence, model_name]

# Convert predictions to a pandas DataFrame and save as CSV
df_predictions = pd.DataFrame(
    highest_confidence_predictions,
    columns=["Image Path", "Label", "Confidence", "Model"],
)
df_predictions.to_csv("highest_confidence_predictions.csv", index=False)


# %%
