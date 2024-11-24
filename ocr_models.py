import easyocr
import os
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
import torch
import torch.nn.functional as F
import numpy as np


class OCRModel:
    def __init__(
        self,
    ):
        pass

    def predict(
        self,
    ):
        return []

    def load_images(self, folder_path):
        self.images = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith(".png")
        ]

    def predict_images(
        self,
    ):
        data = []
        for i in self.images:
            data.append(self.predict(i))
        return data


class EasyOCR(OCRModel):
    def __init__(self, name="EasyOCR", configuration={}):
        self.configuration = configuration
        self.name = name
        self.model = easyocr.Reader(["en"])

    def output_transformation(self, prediction):
        if len(prediction) == 0:
            return [None, 0]
        elif len(prediction) > 1:
            longest_element = max(prediction, key=lambda item: len(item[1]))
            return [longest_element[1], longest_element[2]]
        else:
            return [prediction[0][1], prediction[0][2]]

    def predict(self, image_path):
        return [image_path] + self.output_transformation(
            self.model.readtext(image_path)
        )


class TrOCR(OCRModel):
    def __init__(self, name="TrOCR", model_size="large", configuration={}):
        self.name = name
        if model_size == "large":
            model_name = "microsoft/trocr-large-handwritten"
        else:
            model_name = "microsoft/trocr-base-handwritten"
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        self.configuration = configuration

    def input_transforamtion(self, image_path):
        return Image.open(image_path).convert("RGB")

    def predict(self, image_path):
        image = self.input_transforamtion(image_path)
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
        generated_ids = self.model.generate(
            pixel_values, output_scores=True, return_dict_in_generate=True
        )
        generated_text = self.processor.batch_decode(
            generated_ids.sequences, skip_special_tokens=True
        )[0]
        concat = torch.cat(generated_ids.scores, dim=0)
        confianza = (
            torch.mean(torch.max(F.softmax(concat, dim=1), dim=1).values)
        ).item()
        return [image_path, generated_text, confianza]


class TrOCRModified(TrOCR):
    def __init__(self, name="TrOCRModified", model_size="large", configuration={}):
        super().__init__(name, model_size, configuration)

    def input_transforamtion(self, image_path):
        image = Image.open(image_path).convert("RGB")

        configuration = self.configuration or {
            "contrast_factor": 2.0,
            "sharpen_factor": 2.0,
            "threshold_factor": 0.8,
            "noise_reduction_size": 3,
            "sharpness_enhancement": True,
            "contrast_enhancement": True,
            "threshold_enhancement": True,
        }

        image = image.convert("L")

        if configuration.get("threshold_enhancement", True):
            np_image = np.array(image)
            threshold_value = np.mean(np_image) * configuration["threshold_factor"]
            np_image = (np_image > threshold_value) * 255
            image = Image.fromarray(np_image.astype(np.uint8))

        if configuration.get("contrast_enhancement", True):
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(configuration["contrast_factor"])

        if configuration.get("noise_reduction_size", 3) > 0:
            image = image.filter(
                ImageFilter.MedianFilter(size=configuration["noise_reduction_size"])
            )

        if configuration.get("sharpness_enhancement", True):
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(configuration["sharpen_factor"])

        return image.convert("RGB")
