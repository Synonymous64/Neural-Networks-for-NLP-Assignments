# Neural Networks for NLP Assignments

A collection of assignments of RPTU University covering fundamental concepts in Natural Language Processing (NLP) and Neural Networks. This repository contains practical implementations of various NLP techniques, from basic language classification to advanced tokenization methods.

## Installation

### Requirements
- Python 3.8+
- NumPy
- Pandas
- Scikit-learn
- PyTorch (for neural network exercises)
- Jupyter Notebook
- regex (for advanced regex tokenization)

### Setup

```bash
# Clone the repository
git clone https://github.com/Synonymous64/Neural-Networks-for-NLP-Assignments.git
cd Neural-Networks-for-NLP-Assignments

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install numpy pandas scikit-learn torch jupyter regex
```

---

## Project Structure

```
Neural-Networks-for-NLP-Assignments/
│
├── README.md                              # This file
│
└── Assignment 1/
    │
    ├── Exercise 1.1/
    │   └── Task1_Linear_and_Nonlinear_Models.md
    │
    ├── Exercise 1.2/
    │   └── tutorial_1.ipynb
    │
    └── Exercise 1.3/
        └── BPE/
            ├── base.py                    # Helper functions and base class
            ├── basic.py                   # BasicTokenizer implementation
            ├── regex_tokenizer.py         # RegexTokenizer with regex splitting
            ├── main.ipynb                 # Testing notebook
            ├── sample_text.txt            # Sample training text
            └── models/
                ├── basic.model            # Trained basic tokenizer
                ├── basic.vocab            # Basic tokenizer vocabulary
                ├── regex.model            # Trained regex tokenizer
                └── regex.vocab            # Regex tokenizer vocabulary

└── Assignment 2/
    │
    ├── Exercise 2 Question/
    │   └── main1.tex
    │
    └── Tutorial 2/
        ├── resumes_test.csv
        ├── resumes_train.csv
        ├── resumes.csv
        ├── test_word2vec.py
        ├── tutorial2_task1.py
        ├── tutorial2_task2.py
        ├── tutorial2_task3.py
        ├── tutorial2.ipynb
        └── tutorial2.txt
```


## Author

Neural Networks for NLP Assignments
Student: Prajwal Vilas Urkude

---

## License

This repository is for educational purposes.

---

**Last Updated:** December 7, 2025
