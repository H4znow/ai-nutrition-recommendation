## Data Generation Strategy

1. **User Profile Creation**  
   We generate synthetic user data by sampling basic anthropometric and health variables:
   - **Weight (kg)**: Uniformly in [50, 100], then rounded to two digits after the coma.
   - **Height (cm)**: Uniformly in [150, 200].
   - **BMI**: Computed from weight and height.
   - **Age (years)**: Randomly in [18, 60].
   - **BMR** (Basal Metabolic Rate): Mifflin‐St Jeor formula for males, using the sampled weight, height, and age.
   - **PAL** (Physical Activity Level): Uniformly in [1.2, 2.0].
   - **Medical Conditions**: Three binary flags (CVD, T2D, iron deficiency), each with a small probability (10%).

2. **Daily Energy Intake Target**  
   The user’s **target EI** is computed as `BMR * PAL * D`, where `D` adjusts for BMI category:
   - If BMI < 18.5, \(D = 1.1\) (to encourage weight gain).
   - If BMI > 25, \(D = 0.9\) (to promote weight loss).
   - Otherwise, \(D = 1.0\).

3. **Macro Ranges**  
   Each user’s target EI is translated into ranges for **protein, carbs, fat, and saturated fats (SFA)** by applying standard guideline percentages, then converting to grams:
   - Protein: 10–35% of total energy (\(4 \text{ kcal/g}\)).
   - Carbs: 45–65% of total energy (\(4 \text{ kcal/g}\)).
   - Fat: 20–35% of total energy (\(9 \text{ kcal/g}\)).
   - SFA: 0–10% of total energy (\(9 \text{ kcal/g}\)).  
   This yields per‐user min/max macros in grams.

4. **Meal Plan Generation**  
   We create a 6‐meal sequence per user. Each meal is an integer class label \([0,9]\). The distribution of **high‐calorie** (classes 0–4) vs. **low‐calorie** (classes 5–9) meals depends on BMI:
   - Underweight (BMI<18.5): 80% high‐calorie, 20% low‐calorie.
   - Overweight (BMI>25): 20% high‐calorie, 80% low‐calorie.
   - Normal weight: Uniform random classes.

5. **Final Dataset**  
   We assemble all features (weight, height, BMI, …), the 6 meal classes, min/max macros, and target EI into a single dataset. The data is normalized before being used in training.


## Model Architecture

1. **Encoder**  
   - Fully connected layers (`fc1` and `fc2`) with ReLU activation transform the 8 input features into a hidden representation.
   - Two linear layers output **μ** and **log(σ²)** for the latent variable distribution (latent dimension chosen in code).

2. **Decoder**  
   - Latent vector \(z\) is projected into a hidden dimension, then fed into a 2‐layer **GRUCell** unrolled for 6 time steps.
   - At each time step, the decoder predicts:
     1. **Meal class logits** (10 classes).
     2. **Energy** contribution (scalar) at that step.
     3. **Macros** (protein, carbs, fat, SFA) contribution.
   - The final outputs are the sequence of meal logits, the summed energy over 6 steps, and accumulated macros.

3. **Training Setup**  
   We combine several losses:
   - **KLD** (KL‐Divergence) to keep the latent distribution close to \(\mathcal{N}(0, I)\).
   - **Meal Class Loss** (cross‐entropy) comparing the predicted logits to the ground‐truth meal labels.
   - **Energy Loss** (e.g., MSE) comparing the summed predicted daily energy to the target EI.
   - **Macro Loss** (e.g., penalty for deviating from min/max macro range).

   The total loss is `L_macro + L_energy + L_kld + L_mc`.

4. **Training / Validation Split**  
   We shuffle and split data into 70% for training, 15% for validation, 15% for testing. Batch training proceeds with Adam optimizer, and we track average loss per epoch.


## Training Result Interpretation

- **Loss Trend**  
  Over 500 epochs, both **training (blue)** and **validation (orange)** losses fluctuate in a narrow band around ~7.04–7.09. There is no strong downward trend, indicating the model has essentially **plateaued** given its current architecture and hyperparameters.

- **Convergence and Overfitting**  
  There is not much overlap between validation and training loss, suggesting **no severe overfitting** (the validation loss does not diverge from the training loss). Instead, it appears the VAE has stabilized around a certain solution without dramatic improvement in later epochs.

- **Fluctuations**  
  Because multiple loss terms are added (including KL‐Divergence, cross‐entropy, etc.), some oscillation is expected for a [variation autencoder](https://hunterheidenreich.com/posts/modern-variational-autoencoder-in-pytorch/#training-and-validation:~:text=writer%3Dwriter)-,TensorBoard%20Visualization,-%23). The model regularly samples different latent variables (via the reparameterization trick), so small, constant fluctuations are normal.

At the end, results are not satisfactory. The model does not seem to learn well. Or the randomly generated data is not representative enough of real-world data. The model is not able to learn the underlying distribution of the data, leading to poor performance.