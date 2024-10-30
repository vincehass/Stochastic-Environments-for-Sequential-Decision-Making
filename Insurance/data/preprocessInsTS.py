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
    insurance_pricing_ts.save_dataset("./Insurance/data/insurance_pricing_timeseries_dataset.json")

    # Print a few examples
    with open("./Insurance/data/insurance_pricing_timeseries_dataset.json", 'r') as f:
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

# # Usage example
# if __name__ == "__main__":
#     insurance_pricing_ts = InsurancePricingTimeSeries(num_customers=5, num_time_steps=12)
#     insurance_pricing_ts.save_dataset_as_txt("insurance_pricing_timeseries_dataset.txt")

#     # Print a few examples
#     with open("insurance_pricing_timeseries_dataset.txt", 'r') as f:
#         lines = f.readlines()
#         print("Sample entries from the time series dataset:")
#         for line in lines[:50]:  # Show the first 50 lines as a sample
#             print(line.strip())