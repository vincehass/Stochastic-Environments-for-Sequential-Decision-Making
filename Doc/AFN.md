# Related Work:Adversarial Flow Networks (AFNs)

In Adversarial Flow Networks (AFNs), the training of dynamic stochastic environments revolves around ensuring that the flow of probabilities over trajectories satisfies certain balance conditions like **Trajectory Balance (TB)** or **Detailed Balance (DB)**. Below, I'll outline the mathematical details related to the loss functions and how the probability of the next state, given an action, is calculated in these stochastic settings.

### 1. **Trajectory Balance (TB) Loss Function**

The **Trajectory Balance (TB)** loss ensures that the flow assigned to each trajectory matches the reward at the terminal state. Let’s denote:

- \( \tau = (s_0, a_0, s_1, a_1, \dots, s_T) \) as a trajectory, where \( s_i \) are states and \( a_i \) are actions.
- \( P\_\theta(\tau) \) is the probability of generating trajectory \( \tau \) under policy \( \theta \).
- \( R(s_T) \) is the reward obtained at the terminal state \( s_T \).
- \( Z\_\theta \) is the normalization constant (partition function).

The **Trajectory Balance loss** is given by:

\[
\mathcal{L}_{TB}(\theta) = \mathbb{E}_{\tau \sim P*\theta(\tau)} \left[ \left( \log P*\theta(\tau) - \log Z\_\theta - \log R(s_T) \right)^2 \right]
\]

Where:

- \( P*\theta(\tau) = P(s_0) \prod*{t=0}^{T-1} P*\theta(a_t | s_t) P(s*{t+1} | s_t, a_t) \)
- \( \log P\_\theta(\tau) \) is the log-probability of trajectory \( \tau \).
- \( \log Z\_\theta \) normalizes the flow, and \( \log R(s_T) \) connects the reward at the terminal state to the trajectory probability.

The loss minimizes the difference between the flow through the trajectory and the terminal reward, ensuring that the assigned flows reflect the desired reward structure.

### 2. **Detailed Balance (DB) Loss Function**

The **Detailed Balance (DB)** condition ensures that for any pair of connected states, the flow into a state matches the flow out of the state. In mathematical terms, for any transition \( s \to s' \):

\[
P(s) P*\theta(s'|s) = P(s') P*\theta(s|s')
\]

The **DB loss function** is:

\[
\mathcal{L}_{DB}(\theta) = \mathbb{E}_{(s, s')} \left[ \left( \log P(s) + \log P_\theta(s'|s) - \log P(s') - \log P_\theta(s|s') \right)^2 \right]
\]

This ensures that the forward and backward transitions between states respect the balance of flows, adjusted dynamically according to the environment’s stochastic behavior.

### 3. **Probability of the Next State Given Action \( P(s' | s, a) \)**

In stochastic environments, the transition probability to the next state \( s' \) given the current state \( s \) and action \( a \) is typically modeled as a conditional probability distribution. Let’s denote this as:

\[
P(s' | s, a)
\]

This probability can be learned or predefined based on the environment's stochastic dynamics. In AFNs, this transition probability could be a parametric model \( P\_\theta(s' | s, a) \), where \( \theta \) are the learnable parameters.

The transition probability is usually computed using a softmax or similar function over potential next states. For instance, if we use a neural network to parameterize the probability of the next state:

\[
P*\theta(s' | s, a) = \frac{\exp(f*\theta(s, a, s'))}{\sum*{s''} \exp(f*\theta(s, a, s''))}
\]

Where \( f\_\theta(s, a, s') \) is the output of a neural network that scores the transition from \( s \) to \( s' \) under action \( a \), and the denominator sums over all possible next states \( s'' \).

### 4. **Combining Probability and Reward for Flow Updates**

The flow through a state-action pair in AFNs combines the transition probabilities and rewards. For each state-action pair \( (s, a) \), the flow is updated by balancing the incoming and outgoing flows under the stochastic dynamics:

\[
\text{Flow}(s, a) = P(s) P*\theta(a | s) \sum*{s'} P\_\theta(s' | s, a)
\]

This flow must match the expected reward under the given trajectory, which is encoded by the terminal reward function \( R(s_T) \). The objective is to adjust the parameters \( \theta \) so that the flows through all state-action pairs satisfy the balance conditions imposed by the loss function (TB or DB).

### 5. **Training in Stochastic Environments**

The AFNs incorporate stochasticity in the environment by adjusting the flow assignments dynamically, based on observed transitions. During training:

- **Sample trajectories** from the environment based on the current policy \( P\_\theta(a | s) \).
- **Update the parameters** \( \theta \) by minimizing the TB or DB loss functions, ensuring that the flow through each trajectory matches the observed rewards.
- The probabilities \( P(s' | s, a) \) are adapted during training, capturing the stochastic nature of the environment.

### Summary

In AFNs, the training process in dynamic stochastic environments revolves around balancing the flows between states, ensuring that the transition probabilities and rewards respect the desired trajectory structure. The TB loss function ensures that the flow along trajectories matches the terminal rewards, while the DB loss ensures that transitions between states respect detailed balance. The transition probabilities \( P(s' | s, a) \) are parameterized, typically using neural networks, and updated during training to adapt to the stochasticity of the environment.
