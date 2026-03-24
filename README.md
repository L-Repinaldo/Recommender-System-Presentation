# Recommender System Presentation

A comprehensive educational presentation on recommender systems, featuring implementations of two fundamental recommendation algorithms: **Collaborative Filtering** and **Content-Based Filtering**.

## Overview

This repository contains Jupyter notebooks and supporting code that demonstrate how recommender systems work. Whether you're new to machine learning or looking to deepen your understanding of recommendation algorithms, this project provides hands-on examples and clear explanations.

## Project Structure

```
Recommender-System-Presentation/
├── CollaborativeFiltering/
│   ├── lab.ipynb              # Collaborative Filtering implementation
│   ├── src/                   # Supporting source code
│   └── datasets/              # Data files for experiments
├── ContentBasedFiltering/
│   ├── lab.ipynb              # Content-Based Filtering implementation
│   ├── src/                   # Supporting source code
│   └── datasets/              # Data files for experiments
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Topics Covered

### Collaborative Filtering
Discover how to build recommendations based on user behavior and preferences. This notebook demonstrates how similar users or items can help predict what a user might like.

- **Key Concepts**: User-item interactions, similarity metrics, matrix factorization
- **Location**: `CollaborativeFiltering/lab.ipynb`

### Content-Based Filtering
Learn how to recommend items based on item features and user preferences. This approach analyzes item characteristics to make personalized recommendations.

- **Key Concepts**: Feature extraction, similarity scoring, user profiles
- **Location**: `ContentBasedFiltering/lab.ipynb`

## Requirements

This project requires Python and the following dependencies:

- **matplotlib** (3.10.8) - Visualization library
- **numpy** (2.4.3) - Numerical computing
- **pandas** (3.0.1) - Data manipulation and analysis
- **scikit-learn** (1.8.0) - Machine learning library

## Installation

1. Clone the repository:
```bash
git clone https://github.com/L-Repinaldo/Recommender-System-Presentation.git
cd Recommender-System-Presentation
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Each filtering approach has its own Jupyter notebook with detailed implementations:

1. **Open Collaborative Filtering notebook**:
```bash
jupyter notebook CollaborativeFiltering/lab.ipynb
```

2. **Open Content-Based Filtering notebook**:
```bash
jupyter notebook ContentBasedFiltering/lab.ipynb
```

Run the cells sequentially to see how each algorithm works, from data loading through model evaluation.

## Technology Stack

- **84.1%** Jupyter Notebooks - Interactive educational content
- **15.9%** Python - Core implementation logic
- **Scikit-Learn** - Machine learning algorithms
- **Pandas & NumPy** - Data manipulation
- **Matplotlib** - Data visualization

## Learning Outcomes

By working through this project, you'll understand:

- How collaborative filtering systems make recommendations based on user similarity
- How content-based filtering systems make recommendations based on item features
- The trade-offs between different recommendation approaches
- How to implement and evaluate recommender systems

## Getting Started

Start with the Collaborative Filtering notebook if you're new to recommender systems, then explore Content-Based Filtering to see how different approaches tackle the same problem.

## Contributing

Feel free to fork this repository and submit improvements, corrections, or additional implementations.

## License

This project is open source and available for educational purposes.

---

**Created by**: L-Repinaldo  
**Last Updated**: 2026-03-24