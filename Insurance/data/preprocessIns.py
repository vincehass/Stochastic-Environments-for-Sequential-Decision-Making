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
