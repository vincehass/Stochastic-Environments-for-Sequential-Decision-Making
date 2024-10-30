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
    insurance_pricing_ts.save_dataframe_as_csv("./Insurance/data/insurance_pricing_timeseries_dataset.csv")

    # Print a preview of the DataFrame
    print("Sample entries from the time series DataFrame:")
    print(df.head(10))
