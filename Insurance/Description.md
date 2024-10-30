# Motif scoring for insurance pricing

Motif scoring for insurance pricing by incorporating customer demographics, competitor pricing, and a scoring system for evaluating pricing strategies.

### Python Code: Adapted for Insurance Pricing

```python
import random
import numpy as np
import json

class SimplifiedInsurancePricing:
    def __init__(self, num_features=5):
        self.num_features = num_features
        self.age_range = (18, 80)
        self.income_range = (20000, 150000)
        self.claim_history_range = (0, 10)
        self.competitor_price_variance = 0.1

        # Pricing strategies to simulate customer response
        self.price_sensitivity = {
            'young_low_income': 0.7,
            'young_high_income': 0.5,
            'old_low_income': 0.6,
            'old_high_income': 0.4,
        }

    def generate_customer_features(self):
        # Generate random customer features
        age = random.randint(self.age_range[0], self.age_range[1])
        income = random.uniform(self.income_range[0], self.income_range[1])
        claim_history = random.randint(self.claim_history_range[0], self.claim_history_range[1])
        competitor_price = np.random.normal(1.0, self.competitor_price_variance)

        return {
            'age': age,
            'income': income,
            'claim_history': claim_history,
            'competitor_price': competitor_price
        }

    def calculate_price_score(self, features):
        score = 0
        reasoning = []

        # Determine age and income segment for sensitivity
        if features['age'] < 40:
            if features['income'] < 50000:
                segment = 'young_low_income'
            else:
                segment = 'young_high_income'
        else:
            if features['income'] < 50000:
                segment = 'old_low_income'
            else:
                segment = 'old_high_income'

        # Calculate score based on segment price sensitivity
        sensitivity = self.price_sensitivity[segment]
        price_adjustment = features['competitor_price'] * sensitivity

        # Base price is influenced by competitor pricing and price sensitivity
        base_price = 1.0  # Example base price
        final_price = base_price * (1 + price_adjustment)

        # Calculate reward-like score (profit margin)
        profit_margin = final_price - (0.5 * features['income'] / 100000)  # Example profit calculation
        score += profit_margin

        reasoning.append(f"Segment: {segment}, price sensitivity: {sensitivity:.2f}")
        reasoning.append(f"Competitor price adjustment: {price_adjustment:.2f}")
        reasoning.append(f"Final price calculated: {final_price:.2f}")
        reasoning.append(f"Estimated profit margin: {profit_margin:.2f}")

        # Normalize score to be between 0 and 1
        normalized_score = min(max(score / 5, 0), 1)

        return normalized_score, reasoning

    def generate_dataset(self, num_samples=1000):
        dataset = []
        for _ in range(num_samples):
            features = self.generate_customer_features()
            score, reasoning = self.calculate_price_score(features)
            dataset.append({
                'features': features,
                'score': score,
                'reasoning': reasoning
            })
        return dataset

    def save_dataset(self, filename, num_samples=1000):
        dataset = self.generate_dataset(num_samples)
        with open(filename, 'w') as f:
            json.dump(dataset, f, indent=2)
        print(f"Dataset saved to {filename}")

# Usage example
if __name__ == "__main__":
    insurance_pricing = SimplifiedInsurancePricing()
    insurance_pricing.save_dataset("insurance_pricing_dataset.json", num_samples=1000)

    # Print a few examples
    with open("insurance_pricing_dataset.json", 'r') as f:
        dataset = json.load(f)

    print("Sample entries from the dataset:")
    for item in dataset[:5]:
        print(f"Features: {item['features']}")
        print(f"Score: {item['score']:.4f}")
        print("Reasoning:")
        for reason in item['reasoning']:
            print(f"- {reason}")
        print()
```

### Breakdown of the Code Changes

1. **Customer Features Generation**:

   - **Age**, **income**, **claim history**, and **competitor pricing** are generated as features.
   - The customer demographics are randomized within specified ranges, capturing different segments of customers.

2. **Pricing Score Calculation**:

   - A customer segment is determined based on age and income.
   - **Price Sensitivity**: The model considers the price sensitivity for different segments (e.g., young with low income tends to be more sensitive to price).
   - **Competitor Pricing Adjustment**: The competitor price influences the final price calculated for the customer.
   - **Profit Margin Calculation**: The score is calculated to reflect the profit margin based on the final price and customer income.
   - This score is normalized to fall between 0 and 1.

3. **Dataset Generation**:
   - A dataset is generated with customer features, scores, and detailed reasoning for each score.
   - The dataset is saved as a JSON file for further use in training and testing insurance pricing models.

### Explanation on Adapting to the TFBIND Environment

- **State Representation**: The customer features (age, income, claim history, competitor pricing) represent the state similar to the nucleotide sequences in TFBIND.
- **Scoring System**: The score calculated here plays the role of binding affinity in the original TFBIND task, but adapted to represent the profit margin influenced by market competition and customer characteristics.
- **Dataset Similarity**: The dataset generated follows the same logic as TFBIND, where customer data and their associated scores can be used in a reinforcement learning setting to train an agent.

This approach effectively adapts the TFBIND sequence generation and scoring logic to insurance pricing, ensuring that customer demographics, competitor pricing, and profit margins are captured in the data generation process. Let me know if you need more details or further assistance!

## Time Series Prediction for Insurance Pricing

We adapt the given insurance pricing script to generate time series data, we will need to:

1. Modify the feature generation to create data points over time.
2. Represent each customer’s interaction or changes over a sequence of time steps.
3. Maintain continuity and correlation across time steps, especially for features like income, competitor price, or claim history.

Here, I will implement the logic to generate time series data for multiple customers, where each data point includes information across several time steps.

### Python Code: Time Series Data Generation for Insurance Pricing

```python
import random
import numpy as np
import json

class InsurancePricingTimeSeries:
    def __init__(self, num_customers=100, num_time_steps=12):
        self.num_customers = num_customers
        self.num_time_steps = num_time_steps
        self.age_range = (18, 80)
        self.income_range = (20000, 150000)
        self.claim_history_range = (0, 10)
        self.competitor_price_variance = 0.1

        # Pricing strategies to simulate customer response
        self.price_sensitivity = {
            'young_low_income': 0.7,
            'young_high_income': 0.5,
            'old_low_income': 0.6,
            'old_high_income': 0.4,
        }

    def generate_initial_customer_features(self):
        # Generate initial features for a customer
        age = random.randint(self.age_range[0], self.age_range[1])
        income = random.uniform(self.income_range[0], self.income_range[1])
        claim_history = random.randint(self.claim_history_range[0], self.claim_history_range[1])
        competitor_price = np.random.normal(1.0, self.competitor_price_variance)

        return {
            'age': age,
            'income': income,
            'claim_history': claim_history,
            'competitor_price': competitor_price
        }

    def update_features_over_time(self, features, time_step):
        # Age increases over time
        features['age'] += 1 / 12  # Age grows monthly, converted to years

        # Income changes based on a random growth factor
        income_growth = np.random.normal(1.01, 0.02)  # Average 1% growth per month with variance
        features['income'] *= income_growth

        # Claim history might change over time, e.g., new claims being filed
        if random.random() < 0.05:  # 5% chance per month of a new claim
            features['claim_history'] += 1

        # Competitor pricing is adjusted based on stochastic fluctuations
        features['competitor_price'] = np.random.normal(1.0, self.competitor_price_variance)

        return features

    def calculate_price_score(self, features):
        score = 0
        reasoning = []

        # Determine age and income segment for sensitivity
        if features['age'] < 40:
            if features['income'] < 50000:
                segment = 'young_low_income'
            else:
                segment = 'young_high_income'
        else:
            if features['income'] < 50000:
                segment = 'old_low_income'
            else:
                segment = 'old_high_income'

        # Calculate score based on segment price sensitivity
        sensitivity = self.price_sensitivity[segment]
        price_adjustment = features['competitor_price'] * sensitivity

        # Base price is influenced by competitor pricing and price sensitivity
        base_price = 1.0  # Example base price
        final_price = base_price * (1 + price_adjustment)

        # Calculate reward-like score (profit margin)
        profit_margin = final_price - (0.5 * features['income'] / 100000)  # Example profit calculation
        score += profit_margin

        reasoning.append(f"Segment: {segment}, price sensitivity: {sensitivity:.2f}")
        reasoning.append(f"Competitor price adjustment: {price_adjustment:.2f}")
        reasoning.append(f"Final price calculated: {final_price:.2f}")
        reasoning.append(f"Estimated profit margin: {profit_margin:.2f}")

        # Normalize score to be between 0 and 1
        normalized_score = min(max(score / 5, 0), 1)

        return normalized_score, reasoning

    def generate_time_series_dataset(self):
        dataset = []
        for customer_id in range(self.num_customers):
            # Initialize customer features
            features = self.generate_initial_customer_features()
            customer_time_series = []

            # Generate data for each time step
            for t in range(self.num_time_steps):
                current_features = features.copy()
                score, reasoning = self.calculate_price_score(current_features)
                customer_time_series.append({
                    'time_step': t,
                    'features': current_features,
                    'score': score,
                    'reasoning': reasoning
                })

                # Update features for the next time step
                features = self.update_features_over_time(features, t)

            dataset.append({
                'customer_id': customer_id,
                'time_series': customer_time_series
            })

        return dataset

    def save_dataset(self, filename):
        dataset = self.generate_time_series_dataset()
        with open(filename, 'w') as f:
            json.dump(dataset, f, indent=2)
        print(f"Time series dataset saved to {filename}")

# Usage example
if __name__ == "__main__":
    insurance_pricing_ts = InsurancePricingTimeSeries(num_customers=5, num_time_steps=12)
    insurance_pricing_ts.save_dataset("insurance_pricing_timeseries_dataset.json")

    # Print a few examples
    with open("insurance_pricing_timeseries_dataset.json", 'r') as f:
        dataset = json.load(f)

    print("Sample entries from the time series dataset:")
    for customer in dataset:
        print(f"Customer ID: {customer['customer_id']}")
        for entry in customer['time_series'][:3]:  # Show the first 3 time steps for each customer
            print(f"  Time Step: {entry['time_step']}")
            print(f"  Features: {entry['features']}")
            print(f"  Score: {entry['score']:.4f}")
            print("  Reasoning:")
            for reason in entry['reasoning']:
                print(f"  - {reason}")
            print()
```

### Explanation of Adaptation to Time Series

1. **Time Series Generation**:

   - For each customer, we generate initial features (`age`, `income`, `claim_history`, `competitor_price`).
   - We simulate data across multiple time steps (e.g., 12 months) for each customer.

2. **Updating Features Over Time**:

   - **Age**: Increases gradually over each month (`1/12` years per month).
   - **Income**: Adjusted by a random growth factor to simulate realistic monthly changes.
   - **Claim History**: Has a 5% chance of increasing each month.
   - **Competitor Pricing**: Random fluctuations are added at each time step to simulate competitive market changes.

3. **Score Calculation**:

   - The logic for calculating the price score is similar to the previous script, where features influence the pricing strategy.
   - A score and detailed reasoning are provided for each time step to understand how features like income, age, and competitor price impact the insurance pricing.

4. **Data Structure**:
   - Each customer has a time series dataset with information across multiple time steps.
   - Each time step entry includes the customer features at that time, the calculated score, and a reasoning breakdown.

This code allows you to simulate insurance pricing over time, giving insights into how customer characteristics and market conditions evolve and affect pricing decisions. Let me know if you need more details or modifications!

## Time Series Dataframe for Insurance Pricing

We generates a pandas DataFrame containing time series data for insurance pricing. Each row in the DataFrame represents the data for a specific customer at a specific time step.

This structure allows for easier analysis and manipulation of time series data using pandas.

### Python Code: Generating Pandas Time Series DataFrame

```python
import random
import numpy as np
import pandas as pd

class InsurancePricingTimeSeries:
    def __init__(self, num_customers=100, num_time_steps=12):
        self.num_customers = num_customers
        self.num_time_steps = num_time_steps
        self.age_range = (18, 80)
        self.income_range = (20000, 150000)
        self.claim_history_range = (0, 10)
        self.competitor_price_variance = 0.1

        # Pricing strategies to simulate customer response
        self.price_sensitivity = {
            'young_low_income': 0.7,
            'young_high_income': 0.5,
            'old_low_income': 0.6,
            'old_high_income': 0.4,
        }

    def generate_initial_customer_features(self):
        # Generate initial features for a customer
        age = random.randint(self.age_range[0], self.age_range[1])
        income = random.uniform(self.income_range[0], self.income_range[1])
        claim_history = random.randint(self.claim_history_range[0], self.claim_history_range[1])
        competitor_price = np.random.normal(1.0, self.competitor_price_variance)

        return {
            'age': age,
            'income': income,
            'claim_history': claim_history,
            'competitor_price': competitor_price
        }

    def update_features_over_time(self, features, time_step):
        # Age increases over time
        features['age'] += 1 / 12  # Age grows monthly, converted to years

        # Income changes based on a random growth factor
        income_growth = np.random.normal(1.01, 0.02)  # Average 1% growth per month with variance
        features['income'] *= income_growth

        # Claim history might change over time, e.g., new claims being filed
        if random.random() < 0.05:  # 5% chance per month of a new claim
            features['claim_history'] += 1

        # Competitor pricing is adjusted based on stochastic fluctuations
        features['competitor_price'] = np.random.normal(1.0, self.competitor_price_variance)

        return features

    def calculate_price_score(self, features):
        score = 0
        reasoning = []

        # Determine age and income segment for sensitivity
        if features['age'] < 40:
            if features['income'] < 50000:
                segment = 'young_low_income'
            else:
                segment = 'young_high_income'
        else:
            if features['income'] < 50000:
                segment = 'old_low_income'
            else:
                segment = 'old_high_income'

        # Calculate score based on segment price sensitivity
        sensitivity = self.price_sensitivity[segment]
        price_adjustment = features['competitor_price'] * sensitivity

        # Base price is influenced by competitor pricing and price sensitivity
        base_price = 1.0  # Example base price
        final_price = base_price * (1 + price_adjustment)

        # Calculate reward-like score (profit margin)
        profit_margin = final_price - (0.5 * features['income'] / 100000)  # Example profit calculation
        score += profit_margin

        reasoning.append(f"Segment: {segment}, price sensitivity: {sensitivity:.2f}")
        reasoning.append(f"Competitor price adjustment: {price_adjustment:.2f}")
        reasoning.append(f"Final price calculated: {final_price:.2f}")
        reasoning.append(f"Estimated profit margin: {profit_margin:.2f}")

        # Normalize score to be between 0 and 1
        normalized_score = min(max(score / 5, 0), 1)

        return normalized_score, reasoning

    def generate_time_series_dataframe(self):
        records = []

        for customer_id in range(self.num_customers):
            # Initialize customer features
            features = self.generate_initial_customer_features()

            # Generate data for each time step
            for t in range(self.num_time_steps):
                current_features = features.copy()
                score, reasoning = self.calculate_price_score(current_features)

                # Record data for this customer at this time step
                records.append({
                    'customer_id': customer_id,
                    'time_step': t,
                    'age': current_features['age'],
                    'income': current_features['income'],
                    'claim_history': current_features['claim_history'],
                    'competitor_price': current_features['competitor_price'],
                    'score': score,
                    'reasoning': "; ".join(reasoning)
                })

                # Update features for the next time step
                features = self.update_features_over_time(features, t)

        # Create a DataFrame from the records
        df = pd.DataFrame(records)
        return df

    def save_dataframe_as_csv(self, filename):
        df = self.generate_time_series_dataframe()
        df.to_csv(filename, index=False)
        print(f"Time series DataFrame saved to {filename}")

# Usage example
if __name__ == "__main__":
    insurance_pricing_ts = InsurancePricingTimeSeries(num_customers=5, num_time_steps=12)
    df = insurance_pricing_ts.generate_time_series_dataframe()

    # Save DataFrame to CSV file
    insurance_pricing_ts.save_dataframe_as_csv("insurance_pricing_timeseries_dataset.csv")

    # Print a preview of the DataFrame
    print("Sample entries from the time series DataFrame:")
    print(df.head(10))
```

### Explanation of Changes

1. **DataFrame Generation**:

   - A new method named `generate_time_series_dataframe()` is added to generate a pandas DataFrame.
   - Each row represents a data point for a specific customer at a particular time step.

2. **Data Structure**:

   - Each record contains the following fields:
     - `customer_id`: The ID of the customer.
     - `time_step`: The time step in the series.
     - `age`, `income`, `claim_history`, `competitor_price`: The features of the customer.
     - `score`: The calculated pricing score.
     - `reasoning`: A detailed explanation of how the score was computed.

3. **Saving DataFrame**:

   - The method `save_dataframe_as_csv()` is provided to save the DataFrame to a `.csv` file.
   - The data is saved in a standard CSV format, making it suitable for further analysis.

4. **Previewing the DataFrame**:
   - The script prints the first 10 entries of the generated DataFrame for inspection.

### Example Preview of the DataFrame

The output will look something like:

```
   customer_id  time_step        age        income  claim_history  competitor_price     score                                          reasoning
0            0          0  25.000000   35367.1234             1          0.9567   0.2765   Segment: young_low_income, price sensitivity: 0.70; Competitor price adjustment: 0.67; Final price calculated: 1.67; Estimated profit margin: 1.35
1            0          1  25.083333   35674.5678             1          1.0212   0.3074   Segment: young_low_income, price sensitivity: 0.70; Competitor price adjustment: 0.71; Final price calculated: 1.71; Estimated profit margin: 0.89
...
```

This DataFrame format provides a convenient way to store and manipulate time series data for insurance pricing, making it suitable for machine learning models, exploratory data analysis, and other data science workflows. Each row in the DataFrame provides a snapshot of the customer data and pricing outcome at a specific time step, which can be used for training models or analysis.

Let me know if you need further modifications or explanations!
