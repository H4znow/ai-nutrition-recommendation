# VAE-Based AI Nutrition Recommendation System

A deep generative model for personalized diet planning that uses a Variational Autoencoder (VAE) combined with a GRU-based decoder to generate daily meal plans meeting user-specific energy and nutritional requirements. This project is an experimental implementation of the solution proposed by *Ilias Papastratis , Dimitrios Konstantinidis & al.* in their paper [AI nutrition recommendation using a deep generative model and ChatGPT](https://www.nature.com/articles/s41598-024-65438-x)

---

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Model Architecture](#model-architecture)
- [Installation and Setup](#installation-and-setup)
- [Usage](#usage)
- [Current Status](#current-status)
- [Roadmap and Remaining Tasks](#roadmap-and-remaining-tasks)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Introduction

This project implements an AI-based nutrition recommendation system designed to generate personalized meal plans based on individual user profiles. The system uses:

- **Variational Autoencoder (VAE):** Encodes user profile information (e.g., weight, height, BMI, BMR, Physical Activity Level, and medical conditions) into a structured latent space.
- **GRU-Based Decoder:** Generates a sequence of six meals per day (covering breakfast, morning snack, lunch, afternoon snack, dinner, and supper) by decoding the latent representation.
- **Meal Optimizer:** Adjusts the portions to ensure that the total caloric intake aligns with the user's target energy needs.
- **Future Integration with ChatGPT:** To expand the meal database by generating equivalent meals from diverse cuisines and improve meal variety.

---

## Features

- **Personalized Meal Plans:** Generates tailor-made daily meal plans using user-specific data.
- **Deep Generative Modeling:** Uses a VAE to learn a compact and descriptive latent representation of user profiles.
- **Sequential Meal Generation:** Incorporates a GRU-based decoder for generating sequential meal recommendations.
- **Energy and Nutrient Balancing:** Contains an optimizer module that refines meal portions so total calorie and macronutrient targets are met.
- **Modular Code Structure:** Organized project with separate folders for source code, data, and resources.
- **Logging and Evaluation:** Training outputs and loss values are logged to track model performance.

---

## Model Architecture

### 1. Variational Autoencoder (VAE)
- **Encoder:**  
  - Processes normalized user profiles using fully connected layers.
  - Outputs a latent distribution by computing the mean (μ) and log-variance.
  - Uses the reparameterization trick (z = μ + σ · ε) to sample the latent vector.
  
- **Decoder (GRU-based):**  
  - Receives the latent vector to initialize a GRU.
  - Sequentially generates six meal outputs (one per meal time).
  - Outputs include raw logits for meal classification, predicted energy, and macronutrient estimates.

### 2. Meal Optimizer
- Adjusts portion sizes based on the predicted energy intake against the target energy requirement.
- Applies a correction factor (based on user parameters such as BMR, PAL, and BMI) to zero out the caloric difference.

### 3. Loss Functions
The training objective combines several loss components:
- **Cross-Entropy Loss:** For the meal classification task.
- **Kullback–Leibler Divergence (KL Divergence) Loss:** To regularize the latent space.
- **Energy Intake Loss:** Mean squared error between predicted and target energy.
- **Macronutrient Penalty Loss:** Penalizes deviations from nutritional guidelines for protein, carbohydrates, fat, and saturated fat.
- **Total Loss:** Sum of the above losses to guide model training.

---

## Installation and Setup

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/H4znow/ai-nutrition-recommendation.git
   cd ai-nutrition-recommendation
   ```

2. **Install Dependencies:**

   The project dependencies are listed in `requirements.txt`. Install them using:

   ```bash
   pip install -r requirements.txt
   ```

3. **Project Structure Overview:**

   - `src/`: Contains the model definitions, training scripts, and helper functions.
   - `data/`: Houses the data files (e.g., synthetic user profiles or nutrition plans) used for training and testing.
   - `ressources/`: Additional resources such as configuration files or auxiliary scripts.
   - `readme.md`: This README file.
   - `requirements.txt`: Python package dependencies.

---

## Usage

You can run the main notebook at `src/models/notebooks/model.ipynb` to generate artificial data and train the model.

## Current Status

### What Has Been Completed

- **Core Model Implementation:**
  - Developed the VAE to encode user profiles.
  - Implemented the GRU-based decoder to generate a sequence of meals.
  - Integrated a meal optimizer to adjust portions ensuring matching energy intake.
- **Loss Functions:**
  - Combined meal classification, KL divergence, energy intake, and macronutrient penalty losses.
- **Synthetic Data Generation:**
  - Basic scripts are available for generating synthetic user profiles and meal plans.
- **Logging and Basic Evaluation:**
  - Training logs showing loss reduction over epochs.
  - Preliminary evaluation metrics recorded during model training.
- **Modular Codebase:**
  - Project organized into logical modules making future extensions straightforward.

### What Is Remaining (Future Work)

- **Data Integration and Validation:**
  - Integrate real-world datasets beyond the current synthetic or protein-based dataset.
  - Enhance data preprocessing and validation routines.
- **ChatGPT Integration:**
  - Implement an interface to connect with ChatGPT for generating equivalent meals from diverse cuisines.
  - Map ChatGPT’s output into the project’s meal structure (CSV format conversion and compatibility).
- **User Interface Development:**
  - Develop a web or mobile front-end for user profile input and real-time meal plan display.
- **Evaluation and User Studies:**
  - Perform comprehensive evaluations using both synthetic and real user data.
  - Conduct user studies to capture feedback on meal plan quality and adherence.
- **Documentation and Testing:**
  - Expand documentation with usage examples, API references, and detailed instructions.
  - Add unit tests and integration tests for core modules.
- **Deployment Readiness:**
  - Containerize the application using Docker.
  - Set up CI/CD pipelines for automated testing and deployment.

---

## Roadmap and Remaining Tasks

| Task Category           | Tasks                                                      | Status            |
|-------------------------|------------------------------------------------------------|-------------------|
| **Data & Preprocessing**| Integrate additional real-world nutrition datasets         | Pending           |
|                         | Enhance data validation and preprocessing pipelines         | In Progress       |
| **ChatGPT Integration** | Develop API bridge for ChatGPT-based meal generation         | Not Started       |
|                         | Map generated meals into existing meal plan structure         | Not Started       |
| **UI/UX**               | Develop a web or mobile interface for end-users              | Not Started       |
| **Evaluation**          | Comprehensive performance evaluation on real user data       | Pending           |
|                         | User testing and feedback collection                         | Not Started       |
| **Documentation**       | Expand documentation and add detailed API usage examples       | In Progress       |
| **Testing & Deployment**| Write unit/integration tests for core modules                  | Not Started       |
|                         | Dockerize the application and set up CI/CD pipelines           | Not Started       |

*This README serves as a snapshot of the current progress and future directions of the project. Revisions and improvements are ongoing as the project evolves.*

