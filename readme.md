**VAE-Based AI Nutrition Recommendation System**  
*A Deep Generative Model for Personalized Diet Planning*

## **Table of Contents**
- [Introduction](#introduction)
- [Features](#features)
- [Model Architecture](#model-architecture)
- [Results](#results)

---

## **Introduction**
This project implements an AI-based nutrition recommendation system that uses a Variational Autoencoder (VAE) combined with a GRU-based decoder to generate personalized daily meal plans. The goal is to produce meal plans that meet the user’s energy and nutritional requirements based on their profile (e.g., weight, height, BMI, BMR, PAL, and medical conditions). So far, the core deep generative model has been implemented.

---

## **Features**
- **Personalized Meal Plans:** Generates daily meal plans tailored to user profiles.
- **Deep Generative Modeling:** Uses a VAE to encode user features into a latent space.
- **Meal Sequence Generation:** Employs a GRU-based decoder to produce a sequence of six meals per day.
- **Energy & Nutrient Optimization:** Includes an optimizer module that adjusts meal portions to match the target energy intake.
- **Modular Code Structure:** Organized into separate modules for model definitions, synthetic data generation, and training.

---

## **Model Architecture**
The system comprises three main components:
1. **Variational Autoencoder (VAE)**
   - **Encoder:** Converts user profile features into a latent distribution (mean and log-variance).
   - **Reparameterization Trick:** Samples a latent vector \(z = \mu + \sigma \cdot \epsilon\) to maintain differentiability.
2. **GRU-Based Meal Generator (Decoder)**
   - Projects the latent vector into the GRU hidden space.
   - Generates a sequence of six meals, outputting:
     - Raw logits for meal classification.
     - Predicted energy for each meal.
     - Predicted macronutrient values.
3. **Meal Optimizer**
   - Adjusts the energy portions of each meal to ensure that the total energy intake matches the user’s target.

## **Results**
During training, the model minimizes a combined loss function that includes:
- Meal classification loss (cross-entropy)
- Kullback–Leibler divergence loss for latent regularization
- Energy intake loss (mean squared error)
- Macronutrient penalty loss

The average loss per epoch is logged, showing progress toward generating meal plans that satisfy the nutritional guidelines.

---

*Next Steps:* The next phase will focus on implementing data generation and integrating ChatGPT to expand the meal database with diverse cuisine options.