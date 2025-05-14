# FewShotX

FewShotX is a Python package for zero-shot and few-shot learning, designed for text classification and Natural Language Inference (NLI) tasks. The package provides flexible and intuitive classes to implement custom learning approaches using minimal labeled data.

## 🚀 Features

* **DictionaryScorer**: A scoring based on word-counting methods.
* **ZeroShotLearner**: Leverages embedding-based similarity to classify text without labeled training data.
* **ZeroShotNLI**: Applies Natural Language Inference for entailment, contradiction, and neutral detection.
* **FewShotLearner**: Implements few-shot learning with support for training on a limited number of labeled examples.

## 📦 Installation

Install the latest version:

```bash
pip install https://github.com/RenatoVassallo/FewShotX/releases/download/0.1.3/fewshotx-0.1.3-py3-none-any.whl
```

## 📚 Tutorials and Examples

For detailed tutorials and examples, refer to the following notebooks:

* [Dictionary Tutorial](./tutorials/Dictionary_Tutorial.ipynb): Demonstrates word counting methods using spaCy’s `en_core_web_sm` model for tokenization.
* [ZeroShot Tutorial](./tutorials/ZeroShot_Tutorial.ipynb): Demonstrates text classification using the ZeroShotLearner and ZeroShotNLI classes.
* [FewShot Tutorial](./tutorials/FewShot_Tutorial.ipynb): Step-by-step implementation of FewShotLearner for custom text datasets.

## 📖 Additional Resources

For more comprehensive documentation, slides, and research materials, visit the [BSE Forecast NLP repository](https://github.com/RenatoVassallo/BSE-ForecastNLP).

## 📝 License

This project is licensed under the MIT License.

